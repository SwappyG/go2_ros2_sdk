# NOTE: not sure why im getting import warnings, this is working. 
# We're pinned to a very specific version of aiortc, 1.9. 1.11 doesn't work. 
# 1.13 or higher has conflicts with v0.9 of forked aioice. This should be resolved at some point 
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCDataChannel  # type: ignore
import asyncio
import contextlib
import httpx
import logging
import argparse
import typing as t
import sys
import json
import base64

from go2_robot_sdk.webrtc_relay.webrtc_relay_endpoint_go2 import ConnectArgs
from go2_robot_sdk.webrtc_relay.webrtc_relay_endpoint_webrtc import OfferArgs, OfferReply
from go2_robot_sdk.infrastructure.webrtc.data_decoder import WebRTCDataDecoder
from go2_robot_sdk.webrtc_relay.webrtc_relay_exceptions import recreate_and_raise_exception, StateException  # pyright: ignore[reportUnusedImport]
from go2_robot_sdk.domain.entities.robot_data import RobotData
from go2_robot_sdk.domain.entities.robot_config import RobotConfig
from go2_robot_sdk.webrtc_relay.webrtc_relay_client_video_viewer import display_video
import go2_robot_sdk.infrastructure.webrtc.go2_message_parsers as go2_parsers
import go2_robot_sdk.webrtc_relay.voxel_map_viewer as vmv
from go2_robot_sdk.domain.constants.webrtc_topics import RTC_TOPIC 
from go2_robot_sdk.domain.constants.robot_commands import ROBOT_CMD
from go2_robot_sdk.application.utils import command_generator
from go2_robot_sdk.webrtc_relay.webrtc_stats_monitor import WebRTCStatsMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

class WebRTCRelayClient:
    def __init__(
        self, 
        relay_url: str,
        robot_config: RobotConfig,
        on_robot_data: t.Callable[[RobotData], t.Coroutine[None, None, None]],
        on_video_track: t.Callable[[MediaStreamTrack], t.Coroutine[None, None, None]],
        on_lidar_frame: t.Callable[[dict[str, t.Any]], t.Coroutine[None, None, None]],
        receive_video: bool = True,
        receive_lidar: bool = True,
        receive_robot_data: bool = True,
        subscribed_topics: list[str] = None
    ):
        self.url = relay_url
        self.robot_config = robot_config
        self.client = httpx.AsyncClient(timeout=60.0)
        self._on_robot_data = on_robot_data
        self._on_video_track = on_video_track
        self._on_lidar_frame = on_lidar_frame
        self._data_decoder = WebRTCDataDecoder(enable_lidar_decoding=True)
        self._peer_connection = None
        self._peer_datachannel = None
        self._stats_monitor: WebRTCStatsMonitor | None = None
        self._receive_video = receive_video
        self._receive_lidar = receive_lidar
        self._receive_robot_data = receive_robot_data
        self._subscribed_topics = subscribed_topics or []

    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.shutdown()

    async def shutdown(self):
        # Stop stats monitoring
        if self._stats_monitor:
            await self._stats_monitor.stop()
            self._stats_monitor = None
        
        with contextlib.suppress(Exception):
            await self.client.aclose() 

    async def start(self, connect_go2: bool=True):
        logger.debug("webrtc relay client start")
        if connect_go2:
            await self._connect_to_go2()

        self._peer_connection, self._peer_datachannel = await self._create_peer_connection(
            receive_video=self._receive_video,
            receive_lidar=self._receive_lidar,
            receive_robot_data=self._receive_robot_data,
            subscribed_topics=self._subscribed_topics
        )  

    async def change_obstacle_avoid_state(self, enabled: bool):
        """robot sits down on hind legs (like a real dog would)"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling change_obstacle_avoid_state")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=1001,
            parameters={"enable": enabled},
            topic=RTC_TOPIC['OBSTACLE_AVOID'],
        ))

    async def move(self, forward_velocity: float, strafe_velocity: float, rotation_velocity: float):
        """set the robot velocities. Must be sent frequently to maintain velocity, otherwise robot
        will stop moving. If the frequency is too low, there will be janky movement
        """
        if self._peer_datachannel is None:
            raise StateException("call start before calling move")
    
        self._peer_datachannel.send(command_generator.gen_mov_command(
            x=forward_velocity, 
            y=strafe_velocity, 
            z=rotation_velocity, 
            obstacle_avoidance=False,
        ))

    async def gaze(self, roll_angle: float, pitch_angle: float, yaw_angle: float):
        """causes the robot to look towards the specified angles. This will not cause the 
        robot to move its feet. 0,0,0 is looking forward
        """
        if self._peer_datachannel is None:
            raise StateException("call start before calling gaze")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=ROBOT_CMD['Euler'],
            parameters={'x': roll_angle, 'y': pitch_angle, 'z': yaw_angle},
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def stand_up(self):
        """causes the robot to stand up if it's sitting. Does nothing if it's already standing"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling stand_up")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=ROBOT_CMD['StandUp'],
            parameters=None,
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def lie_down_on_belly(self):
        """robot slowly folds legs in to rest on its belly. This is the smoothest way to de-load the 
        motors in prep for turning the robot off"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling lie_down_on_belly")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=ROBOT_CMD['StandDown'],
            parameters=None,
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def sit_on_hind_legs(self):
        """robot sits down on hind legs (like a real dog would)"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling sit_on_hind_legs")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=ROBOT_CMD['Sit'],
            parameters=None,
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def stand_up_from(self):
        """robot stands up from sitting position (gets up from SIT command)"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling stand_up_from")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=ROBOT_CMD['RiseSit'],
            parameters=None,
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def recovery_stand(self):
        """Recovery stand - robot stands up from any position"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling recovery_stand")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=ROBOT_CMD['RecoveryStand'],
            parameters=None,
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def balance_stand(self):
        """Robot performs balance stand (api_id: 1002)"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling balance_stand")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=1002,  # BALANCE_STAND command ID from raw_commands.md
            parameters=None,
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def stop_move(self):
        """Send STOPMOVE command to stop robot movement (api_id: 1003)"""
        if self._peer_datachannel is None:
            raise StateException("call start before calling stop_move")
    
        self._peer_datachannel.send(command_generator.gen_command(
            cmd=1003,  # STOPMOVE command ID from raw_commands.md
            parameters=None,
            topic=RTC_TOPIC['SPORT_MOD'],
        ))

    async def send_json_command(self, command_str: str) -> None:
        """
        Send a raw JSON command to the robot.

        Args:
            command_str: JSON command as a valid JSON string 
                (e.g., '{"type": "msg", "topic": "...", ...}')

        Raises:
            StateException: If peer datachannel is not initialized
            ValueError: If command format is invalid
            json.JSONDecodeError: If command string is invalid JSON

        Usage:
            command_str = '{"type": "msg", "topic": "rt/api/sport/request", "data": {"header": {"identity": {"id": 12345, "api_id": 1004}}, "parameter": ""}}'
        """
        if self._peer_datachannel is None:
            raise StateException("call start before calling send_json_command")

        # Validate it's valid JSON
        try:
            cmd_dict = json.loads(command_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

        # Validate command structure
        try:
            # Check required fields
            if not isinstance(cmd_dict, dict):
                raise ValueError("Command must be a dictionary/JSON object")
            if "type" not in cmd_dict:
                raise ValueError("Command missing 'type' field")
            if "topic" not in cmd_dict:
                raise ValueError("Command missing 'topic' field")
            if "data" not in cmd_dict:
                raise ValueError("Command missing 'data' field")
            if "header" not in cmd_dict["data"]:
                raise ValueError("Command missing 'data.header' field")
            if "identity" not in cmd_dict["data"]["header"]:
                raise ValueError("Command missing 'data.header.identity' field")
            if "api_id" not in cmd_dict["data"]["header"]["identity"]:
                raise ValueError("Command missing 'data.header.identity.api_id' field")
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid command structure: {e}")

        # Send the command JSON string
        self._peer_datachannel.send(command_str)

    async def _connect_to_go2(self):
        logger.info(f"instructing webrtc relay server to connect to the go2 at {self.robot_config=}")
        r = await self.client.post(f"{self.url}/go2/connect", json=ConnectArgs(
            robot_ip=self.robot_config.robot_ip_list[0],
            robot_num=1,  # TODO (swapnil) - pipe this properly
            token=self.robot_config.token
        ).model_dump())
        if r.status_code != 200:
            err_json = r.json()
            logger.warning(f"{r.status_code=} {err_json=}")
            recreate_and_raise_exception(err_json)
            
        logger.info("webrtc server reported successful connection to go2", r.json())

    async def _disconnect_from_go2(self):
        try:
            r = await self.client.post(f"{self.url}/disconnect")
            if r.status_code != 200:
                recreate_and_raise_exception(r.json())
            logger.info("[client] /disconnect:", r.json())
        except Exception as e:
            logger.info("[client] /disconnect failed:", e)

    async def _create_peer_connection(
        self,
        receive_video: bool = True,
        receive_lidar: bool = True,
        receive_robot_data: bool = True,
        subscribed_topics: list[str] = None
    ) -> tuple[RTCPeerConnection, RTCDataChannel]:
        logger.info(f"establishing WebRTC connection to webrtc relay server")
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        peer.on(
            "connectionstatechange", 
            lambda: logger.info(f"webrtc relay client peer connection {peer.connectionState=}")
        )
        peer.on("track", self._on_peer_track)

        # Create the channel here so the OFFER includes m=application (SCTP)
        peer_datachannel = peer.createDataChannel("data")
        peer_datachannel.on("open", lambda : self._on_peer_datachannel_open(peer_datachannel))
        peer_datachannel.on("message", self._on_peer_datachannel_message)
        _peer_transceiver = peer.addTransceiver("video", direction="recvonly")

        # Create offer (no trickle)
        peer_offer = await peer.createOffer()
        await peer.setLocalDescription(peer_offer)
        await self._wait_for_ice_gathering_complete(peer)

        peer_offer_args = OfferArgs(
            sdp=peer.localDescription.sdp,
            type=peer.localDescription.type,
            receive_video=receive_video,
            receive_lidar=receive_lidar,
            receive_robot_data=receive_robot_data,
            subscribed_topics=subscribed_topics or []
        )
        logger.info(f"sending webrtc connection offer to webrtc relay server. {peer_offer_args=}")
        resp = await self.client.post(
            f"{self.url}/webrtc/offer", 
            json=peer_offer_args.model_dump()
        )
        if resp.status_code != 200:
            err_json = resp.json()
            logger.warning(f"webrtc relay client offer failed. {err_json=}")
            recreate_and_raise_exception(err_json)
        
        answer = OfferReply.model_validate(resp.json())
        logger.info(f"received answer from webrtc relay server. {answer=}. Connection established, waiting for data and video channels.")
        await peer.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
        
        # Start WebRTC stats monitoring for client→relay connection
        import os
        debug_stats = os.getenv("DEBUG_WEBRTC_STATS", "false").lower() in ("true", "1", "yes")
        self._stats_monitor = WebRTCStatsMonitor("CLIENT→RELAY", peer, debug=debug_stats)
        await self._stats_monitor.start(interval_seconds=5.0)
        
        return peer, peer_datachannel
          
    def _on_peer_datachannel_open(self, peer_connection_data_channel: RTCDataChannel):
        logger.info(f"datachannel to webrtc relay server is now open. {peer_connection_data_channel=}")
        # if args.send_ping:
        #     logger.info(f"send_pings was set, sending payload thru data channel")
        #     payload = bytes([0x01, 0x02, 0x03, 0x04])
        #     peer_connection_data_channel.send(payload)
        #     logger.info("[data -> GO2] bytes", payload)

    async def _on_peer_datachannel_message(self, data: bytes | str | t.Any):
        try:
            if isinstance(data, bytes):
                logger.debug("got lidar data")
                lidar_frame = await asyncio.to_thread(self._data_decoder.decode_array_buffer, data)
                if lidar_frame is None:
                    logger.warning(f"failed to decode binary message from data_channel")
                    return

                await self._on_lidar_frame(lidar_frame)
                return

            elif isinstance(data, str):
                ret = go2_parsers.parse_datachannel_message(data)
                robot_data = go2_parsers.process_webrtc_message(ret, "0")
                if robot_data:
                    await self._on_robot_data(robot_data)

                return

            else:
                logger.warning(f"got unexpected data type from webrtc relay: {str(type(data))}, {data=}")
                return
            
        except BaseException as exception:
            logger.warning(f"got exception while trying to parse message from webrtc relay data channel. {exception=}")

    async def _on_peer_track(self, track: MediaStreamTrack):
        logger.info(f"received video track from webrtc relay server. {track=}")
        if track.kind != "video":
            logger.info(f"track type was not video, ignoring")
            return
        
        await self._on_video_track(track)

    async def _wait_for_ice_gathering_complete(self, pc: RTCPeerConnection):
        if pc.iceGatheringState == "complete":
            return
        done = asyncio.get_event_loop().create_future()

        def check_state():
            if pc.iceGatheringState == "complete" and not done.done():
                done.set_result(True)
        
        pc.add_listener("icegatheringstatechange", check_state)
        check_state()
        await done


async def keyboard_control_loop(client):
    """Read single-line keyboard commands from stdin in a thread and send move commands.

    Keys:
      w -> up (forward)
      s -> down (back)
      a -> left (strafe left)
      d -> right (strafe right)
      q -> quit the control loop
    """
    print("Keyboard control: w(up), a(left), s(down), d(right), q(quit)")
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                await asyncio.sleep(0.1)
                continue
            key = line.strip().lower()
            if not key:
                continue
            if key == "q":
                print("command: quit")
                break
            if key == "w":
                await client.move(0.5, 0.0, 0.0)
                print("command: up")
            elif key == "s":
                await client.move(-0.5, 0.0, 0.0)
                print("command: down")
            elif key == "a":
                await client.move(0.0, 0.5, 0.0)
                print("command: left")
            elif key == "d":
                await client.move(0.0, -0.5, 0.0)
                print("command: right")
            elif key == "z":
                await client.move(0.0, 0.0, 0.5)
                print("command: rotate left")
            elif key == "c":
                await client.move(0.0, 0.0, -0.5)
                print("command: rotate right")
            elif key == "t":
                # Test JSON command
                custom_cmd = '{"type": "msg", "topic": "rt/api/sport/request", "data": {"header": {"identity": {"id": 12345, "api_id": 1004}}, "parameter": ""}}'
                try:
                    await client.send_json_command(custom_cmd)
                    print("✓ JSON command sent successfully!")
                except Exception as e:
                    print(f"✗ Error: {e}")
            else:
                print(f"unknown command: {key}")
    except asyncio.CancelledError:
        return


async def main(
    relay_url: str, 
    config: RobotConfig,
    on_robot_data: t.Callable[[RobotData], t.Coroutine[None, None, None]], 
    on_video_track: t.Callable[[MediaStreamTrack], t.Coroutine[None, None, None]],
    on_lidar_update: t.Callable[[dict[str, t.Any]], t.Coroutine[None, None, None]],
    receive_video: bool = True,
    receive_lidar: bool = True,
    receive_robot_data: bool = True,
    subscribed_topics: list[str] = None
):
    async with WebRTCRelayClient(
        relay_url=str(relay_url), 
        robot_config=config, 
        on_video_track=on_video_track,
        on_lidar_frame=on_lidar_update,
        on_robot_data=on_robot_data,
        receive_video=receive_video,
        receive_lidar=receive_lidar,
        receive_robot_data=receive_robot_data,
        subscribed_topics=subscribed_topics
    ) as client:
        logger.info("created webrtc relay client, calling start")
        await client.start(True)
        logger.info("created webrtc relay client, started")

        # start keyboard control loop
        keyboard_task = asyncio.create_task(keyboard_control_loop(client))
        try:
            await keyboard_task
        finally:
            keyboard_task.cancel()
            with contextlib.suppress(Exception):
                await keyboard_task

        while True:
            await asyncio.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple PC client for GO2 Pi bridge")
    p.add_argument("--api", default="http://localhost:8000", help="Pi bridge base URL")
    p.add_argument("--robot-ip", default="192.168.12.1", help="GO2 AP IP (optional: call /connect first)")
    p.add_argument("--robot-num", type=int, default=0)
    p.add_argument("--token", default="")
    p.add_argument("--dump-lidar", action="store_true", dest="dump_lidar", help="Write lidar frames to lidar_dump.txt")
    p.add_argument("--send-ping", action="store_true", help="Send a small bytes payload on datachannel open")
    p.add_argument("--disconnect-on-exit", default=True, action="store_true", help="Call /disconnect on exit")
    args = p.parse_args()

    config = RobotConfig(
        robot_ip_list=[args.robot_ip], 
        token=args.token, 
        conn_type="webrtc",
        enable_video=True, 
        decode_lidar=True,
        publish_raw_voxel=True, 
        obstacle_avoidance=True, 
        conn_mode='single'
    )

    TOPICS_TO_SUBSCRIBE_TO = [
        RTC_TOPIC['MULTIPLE_STATE'],
        RTC_TOPIC['SPORT_MOD_STATE'],
        RTC_TOPIC['LOW_STATE'],
        RTC_TOPIC['ULIDAR'],
        RTC_TOPIC['ULIDAR_ARRAY'],
        RTC_TOPIC['ULIDAR_STATE'],
        RTC_TOPIC['ROBOTODOM'],
    ]

    receive_video = True
    receive_lidar = False
    receive_robot_data = True
    subscribed_topics = TOPICS_TO_SUBSCRIBE_TO

    # async def on_video_track(track: MediaStreamTrack):
    #     print(f"Video track received: {track}")

    # async def on_robot_data(robot_data: RobotData):
    #     print(f"Robot data received: {robot_data}")

    
    # asyncio.run(main(
    #     relay_url=args.api,
    #     config=config,
    #     on_robot_data=on_robot_data,
    #     on_video_track=on_video_track,
    #     on_lidar_update=on_lidar_update
    # ))

    display_task: asyncio.Task[None] | None = None
    try:
        async def on_video_track(track: MediaStreamTrack):
            logger.info(f"got video track: {track}")
            global display_task
            if display_task is not None:
                display_task.cancel()
                await display_task

            display_task = asyncio.create_task(display_video(track))        

        async def on_lidar_update(lidar_frame: dict[str, t.Any]):
            print(f"Lidar frame received with {lidar_frame.get('decoded_data', {}).get('face_count', 0)} faces")


        # vmv_viewer = vmv.VoxelMapViewer(flip_winding=False, compute_normals_every=1)
        # ff = open("lidar_dump.txt", mode="w+") if args.dump_lidar else None
        # num_writes = 0
        # try:
        #     vmv_viewer.start()
        #     async def on_lidar_update(lidar_frame: dict[str, t.Any]):
        #         global num_writes
        #         dec = lidar_frame["decoded_data"]
        #         meta = lidar_frame["data"]
        #         positions = dec["positions"]           # np.uint8, length = face_count*12
        #         face_count = int(dec["face_count"])
        #         vmv_viewer.submit_u8(
        #             positions_u8=positions,
        #             face_count=face_count,
        #             resolution=float(meta["resolution"]),
        #             origin_xyz=meta["origin"],
        #         )
        #         if ff and num_writes < 1000:
        #             num_writes += 1
        #             combined = lidar_frame["compressed_metadata"] + lidar_frame["compressed_data"]
        #             b64 = base64.b64encode(combined).decode("ascii")
        #             record = {"frame": b64}
        #             ff.write(json.dumps(record) + "\n")

            # New robot data hook
            # async def on_robot_data(robot_data):
            #     # logger.debug("on robot data")
            #     try:
            #         if robot_data and robot_data.odometry_data:
            #             odom = robot_data.odometry_data
            #             # vmv_viewer.submit_robot_pose(
            #             #     position=odom.position,         # {"x":..,"y":..,"z":..}
            #             #     orientation=odom.orientation,   # {"x":..,"y":..,"z":..,"w":..}
            #             # )
            #             logger.info("Odom position: %s", json.dumps(odom.position))
            #             logger.info("Odom orientation: %s", json.dumps(odom.orientation))
            #     except Exception as e:
            #         logger.warning(f"robot pose update failed: {e}")

        async def on_robot_data(robot_data: RobotData):
            pass

        asyncio.run(main(
            relay_url=args.api, 
            config=config, 
            on_robot_data=on_robot_data, 
            on_video_track=on_video_track, 
            on_lidar_update=on_lidar_update,
            receive_video=receive_video,
            receive_lidar=receive_lidar,
            receive_robot_data=receive_robot_data,
            subscribed_topics=subscribed_topics
        ))
        # finally:
        #     if ff:
        #         ff.close()
        #     vmv_viewer.close()
    finally:
        if display_task is not None:
            display_task.cancel()
            asyncio.wait_for(display_task, timeout=None)
