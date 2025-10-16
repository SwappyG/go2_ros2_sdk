import dataclasses
import asyncio
import logging
from fastapi.requests import Request
from aiortc import RTCPeerConnection, RTCDataChannel, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

from go2_robot_sdk.infrastructure.webrtc.go2_connection import Go2Connection

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class WebRTCRelayAppState:
    media_relay: MediaRelay = MediaRelay()
    go2: Go2Connection | None = None
    relay_rtc_peer_connection: RTCPeerConnection | None = None
    relay_rtc_data_channel: RTCDataChannel | None = None
    go2_video_track: MediaStreamTrack | None = None

    async def close_rtc_relay_connection(self):
        if self.relay_rtc_peer_connection is not None:
            logger.info(f"closing existing rtc peer connection. {self.relay_rtc_peer_connection}")
            try:
                await self.relay_rtc_peer_connection.close()
            except Exception as exception:
                logger.warning(f"failed to close existing relay rtc peer connection. {exception=}")
            finally:
                self.relay_rtc_peer_connection = None

        if self.relay_rtc_data_channel is not None:
            logger.info(f"closing existing rtc peer data channel. {self.relay_rtc_data_channel}")
            try:
                await asyncio.to_thread(self.relay_rtc_data_channel.close)
            except Exception as exception:
                logger.warning(f"failed to close existing relay rtc peer data connection. {exception=}")
            finally:
                self.relay_rtc_data_channel = None 

def get_app_state(request: Request) -> WebRTCRelayAppState:
    return request.app.state.state

