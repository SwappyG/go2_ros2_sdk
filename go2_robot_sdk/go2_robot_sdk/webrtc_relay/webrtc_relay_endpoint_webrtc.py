from aiortc import RTCPeerConnection, RTCDataChannel, RTCConfiguration, RTCSessionDescription  # type: ignore
from fastapi import Depends, APIRouter
import logging
from pydantic import BaseModel
import typing as t

from go2_robot_sdk.webrtc_relay.webrtc_relay_app_state import get_app_state, WebRTCRelayAppState
from go2_robot_sdk.webrtc_relay.webrtc_relay_exceptions import StateException
from go2_robot_sdk.webrtc_relay.webrtc_stats_monitor import WebRTCStatsMonitor

logger = logging.getLogger(__name__)
router = APIRouter()

class OfferArgs(BaseModel):
    sdp: str
    type: str
    receive_video: bool = True
    receive_lidar: bool = True
    receive_robot_data: bool = True
    subscribed_topics: list[str] = []  # Optional: specific topics to subscribe to

class OfferReply(BaseModel):
    sdp: str
    type: str

class UpdateSubscriptionArgs(BaseModel):
    receive_video: bool | None = None
    receive_lidar: bool | None = None
    receive_robot_data: bool | None = None
    subscribed_topics: list[str] | None = None

def _on_datachannel_message(state: WebRTCRelayAppState, message: t.Any):
    """handler for messages inbound from relay'ed webrtc connection"""
    logger.debug(f"relay rtc data channel got {message=}")
    # take reference to go2 in case its modified later
    # go2: Go2Connection = state.go2
    if not state.go2:
        logger.warning(f"go2 has no data_channel connected to send message to")
        return
    
    if isinstance(message, str):
        # Assume this is a json string and forward it
        state.go2.publish_json_str(message)
        return
    
    logger.warning(f"Got unexpected data type in datachannel: {str(type(message))}, {message=}")
    return


def _on_datachannel(state: WebRTCRelayAppState, channel: RTCDataChannel):  # pyright: ignore[reportUnusedFunction]
    logger.info(f"relay_rtc_connection received data channel, {channel.label=}")
    state.relay_rtc_data_channel = channel

    channel.on("open", lambda *_args: logger.info("WebRTC relay data channel connection open"))
    channel.on("message", lambda message: _on_datachannel_message(state, message))
    

@router.post("/offer", response_model=OfferReply)
async def offer(
    sdp: OfferArgs,
    state: WebRTCRelayAppState = Depends(get_app_state)
):

    if state.go2 is None:
        raise StateException("connection to the go2 hasn't been established yet, call /connect first")
    
    await state.close_rtc_relay_connection()

    try:
        logger.info(f"creating new rtc connection to relay data from go2 to caller")
        new_relay_peer_connection = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        state.relay_rtc_peer_connection = new_relay_peer_connection

        # Accept PC-created data channel
        new_relay_peer_connection.on("datachannel", lambda data: _on_datachannel(state, data))

         # Conditionally attach GO2 video based on subscription
        if sdp.receive_video and state.go2_video_track:
            logger.info(f"adding go2 video track to new relay connection")
            new_relay_peer_connection.addTrack(state.media_relay.subscribe(state.go2_video_track))
        else:
            logger.info(f"skipping video track (receive_video={sdp.receive_video})")

         # Store subscription preferences in state
        state.receive_video = sdp.receive_video
        state.receive_lidar = sdp.receive_lidar
        state.receive_robot_data = sdp.receive_robot_data
        state.subscribed_topics = set(sdp.subscribed_topics) if sdp.subscribed_topics else set()

        # SDP handshake
        logger.info(f"relay RTC setting remote description")
        await new_relay_peer_connection.setRemoteDescription(RTCSessionDescription(sdp=sdp.sdp, type=sdp.type))
        logger.info(f"relay RTC creating answer")
        answer = await new_relay_peer_connection.createAnswer()
        
        logger.info(f"relay RTC setting local description")

        # NOTE: in the newer aiortc versions, answer is always a type. But here in 1.9, it can be None
        await new_relay_peer_connection.setLocalDescription(answer)  # type: ignore

        # Start WebRTC stats monitoring for relay→client connection
        # This monitors RELAY→CLIENT (relay sending to client)
        import os
        debug_stats = os.getenv("DEBUG_WEBRTC_STATS", "false").lower() in ("true", "1", "yes")
        stats_monitor_relay_to_client = WebRTCStatsMonitor("RELAY→CLIENT", new_relay_peer_connection, debug=debug_stats)
        await stats_monitor_relay_to_client.start(interval_seconds=5.0)
        state.relay_stats_monitor = stats_monitor_relay_to_client
        
        # Also monitor CLIENT→RELAY (relay receiving from client)
        # The relay's peer connection receives from client, so we can get RTT from remote-inbound-rtp
        stats_monitor_client_to_relay = WebRTCStatsMonitor("CLIENT→RELAY", new_relay_peer_connection, debug=debug_stats)
        await stats_monitor_client_to_relay.start(interval_seconds=5.0)
        state.client_to_relay_stats_monitor = stats_monitor_client_to_relay

        # Re-trigger video to push fresh SPS/PPS for new subscriber
        try:
            if state.go2:
                state.go2.publish("", "on", "vid")
        except Exception as exception:
            logger.warning("Could not re-trigger video:", exception)
        
        return OfferReply(
            sdp=new_relay_peer_connection.localDescription.sdp,
            type=new_relay_peer_connection.localDescription.type,
        )
    
    except Exception as exception:
        logger.warning(f"Failed to create relay rtc sessiondescriptionprotocol (SDP). {exception=}")
        raise StateException(f"Failed to create relay rtc sessiondescriptionprotocol (SDP)") from exception

@router.post("/subscription/update")
async def update_subscription(
    args: UpdateSubscriptionArgs,
    state: WebRTCRelayAppState = Depends(get_app_state)
):
    """Update subscription preferences for the current connection"""
    video_changed = False
    old_video_state = state.receive_video
    
    if args.receive_video is not None:
        state.receive_video = args.receive_video
        video_changed = (old_video_state != args.receive_video)
    
    if args.receive_lidar is not None:
        state.receive_lidar = args.receive_lidar
    if args.receive_robot_data is not None:
        state.receive_robot_data = args.receive_robot_data
    if args.subscribed_topics is not None:
        state.subscribed_topics = set(args.subscribed_topics)
    
    # Update video track if video subscription changed
    if video_changed and state.relay_rtc_peer_connection:
        transceivers = state.relay_rtc_peer_connection.getTransceivers()
        video_transceiver = None
        for transceiver in transceivers:
            if transceiver.kind == "video":
                video_transceiver = transceiver
                break
        
        if state.receive_video:
            if video_transceiver is None and state.go2_video_track:
                # Add video track (lines 78-80 logic)
                logger.info("Adding video track to existing connection")
                state.relay_rtc_peer_connection.addTrack(
                    state.media_relay.subscribe(state.go2_video_track)
                )
            elif video_transceiver:
                # Enable transceiver
                video_transceiver.direction = "recvonly"
                logger.info("Video track enabled")
        else:
            if video_transceiver:
                # Disable transceiver
                video_transceiver.direction = "inactive"
                logger.info("Video track disabled")
    
    return {"status": "updated", "subscriptions": {
        "receive_video": state.receive_video,
        "receive_lidar": state.receive_lidar,
        "receive_robot_data": state.receive_robot_data,
        "subscribed_topics": list(state.subscribed_topics)
    }}
