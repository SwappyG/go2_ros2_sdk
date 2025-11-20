
from aiortc import MediaStreamTrack # type: ignore
import asyncio
from fastapi import HTTPException, Depends, APIRouter
import json
import logging
from pydantic import BaseModel
import typing as t  # pyright: ignore[reportUnusedImport]

from go2_robot_sdk.domain.constants.webrtc_topics import RTC_TOPIC
from go2_robot_sdk.infrastructure.webrtc.go2_connection import Go2Connection, RobotData
from go2_robot_sdk.webrtc_relay.webrtc_relay_app_state import WebRTCRelayAppState, get_app_state 
from go2_robot_sdk.webrtc_relay.webrtc_relay_exceptions import StateException
from go2_robot_sdk.webrtc_relay.webrtc_stats_monitor import WebRTCStatsMonitor
from go2_robot_sdk.webrtc_relay.webrtc_relay_client_video_viewer import display_video


logger = logging.getLogger(__name__)
router = APIRouter()

TOPICS_TO_SUBSCRIBE_TO = [
    RTC_TOPIC['MULTIPLE_STATE'],
    RTC_TOPIC['SPORT_MOD_STATE'],
    RTC_TOPIC['LOW_STATE'],
    RTC_TOPIC['ULIDAR'], 
    RTC_TOPIC['ULIDAR_ARRAY'], 
    RTC_TOPIC['ULIDAR_STATE'],
    RTC_TOPIC['ROBOTODOM'],
]

def _on_go2_message(state: WebRTCRelayAppState, robot_data: RobotData):
    """
    Relay ONLY the parsed object (2nd arg) from GO2 -> PC.
    Serialize to JSON and send to the PC datachannel as TEXT.
    """
    # pc_dc: RTCDataChannel | None = state.relay_rtc_data_channel
    if not state.relay_rtc_data_channel or state.relay_rtc_data_channel.readyState != "open":
        logging.debug(f'got message from go2, but datachannel is not open {robot_data.raw_message=}')
        return

    try:
        # Determine message type and topic
        is_lidar = isinstance(robot_data.raw_message, bytes)
        is_robot_data = isinstance(robot_data.raw_message, str)
        
        # Filter by type
        if is_lidar:
            if not state.receive_lidar:
                return  # Skip lidar data
        elif is_robot_data:
            if not state.receive_robot_data:
                return  # Skip robot data
            
            # Optional: Filter by specific topics
            if state.subscribed_topics:
                try:
                    import json
                    msg_dict = json.loads(robot_data.raw_message)
                    topic = msg_dict.get('topic', '')
                    if topic not in state.subscribed_topics:
                        return  # Skip this topic
                except (json.JSONDecodeError, KeyError):
                    pass  # If we can't parse, send it anyway
        
        # Send the message if it passed filters
        if is_lidar:
            state.relay_rtc_data_channel.send(robot_data.raw_message)
        elif is_robot_data:
            state.relay_rtc_data_channel.send(robot_data.raw_message)
        else:
            print(f"unknown raw type {type(robot_data.raw_message)}")

    except Exception as exception:
        logger.warning(f"Failed to send GO2 message: {exception=}")

def _on_go2_validated(state: WebRTCRelayAppState, topics_to_subscribe_to: list[str]):
    logger.info("on validated called")
    try:
        if state.go2 is not None:
            asyncio.get_running_loop().create_task(state.go2.disableTrafficSaving(True))
            for topic in topics_to_subscribe_to:
                state.go2.data_channel.send(
                    json.dumps({"type": "subscribe", "topic": topic})
                )

            state.go2.publish(RTC_TOPIC['ULIDAR_SWITCH'], 'on')
    except Exception as e:
        logger.error(f"Error in validated callback: {e}")


async def _on_go2_video_track(state: WebRTCRelayAppState, track: MediaStreamTrack, _robot_num: str|int):
    """
    Store the GO2 video track. We'll attach it to a PC RTCPeerConnection
    when the PC calls /offer. We’ll relay via MediaRelay for multi-subscriber safety.
    """
    logger.info(f"received go2 video track, {track=}")
    if state.go2_video_track is not None:
        state.go2_video_track.stop()

    # async def on_video_track(track: MediaStreamTrack):
    #     logger.info(f"got video track: {track}")
    #     global display_task
    #     if display_task is not None:
    #         display_task.cancel()
    #         await display_task

    # state.display_task = asyncio.create_task(display_video(track))
    state.go2_video_track = track

class ConnectArgs(BaseModel):
    robot_ip: str = "192.168.12.1"
    robot_num: int = 0
    token: str = ""
    topics_to_subscribe_to: list[str] = TOPICS_TO_SUBSCRIBE_TO

class ConnectReply(BaseModel):
    robot_ip: str


@router.post("/connect", response_model=ConnectReply)
async def connect(args: ConnectArgs, state: WebRTCRelayAppState = Depends(get_app_state)):
    """
    Connect Raspberry Pi to the GO2 over the AP subnet using your Go2Connection.
    Stores the connection and (optional) video track in app.state.
    """
    if state.go2 is not None:
        raise StateException("Already connected to Go2, call disconnect first before calling connect again")

    go2 = Go2Connection(
        robot_ip=args.robot_ip,
        robot_num=args.robot_num,
        token=args.token,
        on_open=lambda : logger.info("GO2 data channel open"),
        on_message=lambda robot_data: _on_go2_message(state, robot_data),
        on_validated=lambda _robot_id:_on_go2_validated(state, args.topics_to_subscribe_to),
        on_video_frame=lambda track, rn: _on_go2_video_track(state, track, rn),
        decode_lidar=False,
        decode_message=False,
    )
    try:
        await go2.connect()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GO2 connect failed: {e}")

    state.go2 = go2
    
    # Wait for WebRTC connection to be fully established before starting stats monitoring
    # This ensures stats collection will have meaningful data
    max_wait_time = 10.0  # Maximum wait time in seconds
    wait_interval = 0.1   # Check every 100ms
    waited = 0.0
    while go2.pc.connectionState != "connected" and waited < max_wait_time:
        await asyncio.sleep(wait_interval)
        waited += wait_interval
    
    if go2.pc.connectionState != "connected":
        logger.warning(f"GO2 connection state is {go2.pc.connectionState} after {waited:.1f}s, starting stats monitor anyway")
    else:
        logger.info(f"GO2 connection established (state: {go2.pc.connectionState}), starting stats monitor")
    
    # Start WebRTC stats monitoring for GO2→Relay connection
    import os
    debug_stats = os.getenv("DEBUG_WEBRTC_STATS", "false").lower() in ("true", "1", "yes")
    go2_stats_monitor = WebRTCStatsMonitor("GO2→RELAY", go2.pc, debug=debug_stats)
    await go2_stats_monitor.start(interval_seconds=5.0)
    state.go2_stats_monitor = go2_stats_monitor
    
    return ConnectReply(robot_ip=args.robot_ip)


class DisconnectArgs(BaseModel):
    pass

class DisconnectReply(BaseModel):
    pass


@router.post("/disconnect", response_model=DisconnectReply)
async def disconnect(_args: DisconnectArgs, state: WebRTCRelayAppState = Depends(get_app_state)):
    """
    Disconnect from GO2 and tear down any existing PC session.
    """
    # Stop stats monitoring
    if state.go2_stats_monitor:
        await state.go2_stats_monitor.stop()
        state.go2_stats_monitor = None
    
    # Close PC side first
    if state.relay_rtc_peer_connection:
        await state.relay_rtc_peer_connection.close()
        state.relay_rtc_peer_connection = None
        state.relay_rtc_data_channel = None

    # Close GO2
    if state.go2:
        await state.go2.disconnect()
        state.go2 = None
        state.go2_video_track = None

    return
