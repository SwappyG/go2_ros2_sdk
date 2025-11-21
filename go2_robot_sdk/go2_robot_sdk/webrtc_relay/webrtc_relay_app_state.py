import dataclasses
import asyncio
import logging
from typing import TYPE_CHECKING
from fastapi.requests import Request
from aiortc import RTCPeerConnection, RTCDataChannel, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

from go2_robot_sdk.infrastructure.webrtc.go2_connection import Go2Connection

if TYPE_CHECKING:
    from go2_robot_sdk.webrtc_relay.webrtc_stats_monitor import WebRTCStatsMonitor

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class WebRTCRelayAppState:
    media_relay: MediaRelay = dataclasses.field(default_factory=MediaRelay)
    go2: Go2Connection | None = None
    relay_rtc_peer_connection: RTCPeerConnection | None = None
    relay_rtc_data_channel: RTCDataChannel | None = None
    go2_video_track: MediaStreamTrack | None = None
    subscribed_topics: set[str] = dataclasses.field(default_factory=set)
    # WebRTC stats monitoring
    relay_stats_monitor: 'WebRTCStatsMonitor | None' = None  # RELAY→CLIENT
    client_to_relay_stats_monitor: 'WebRTCStatsMonitor | None' = None  # CLIENT→RELAY (from relay's perspective)
    go2_stats_monitor: 'WebRTCStatsMonitor | None' = None  # GO2→RELAY (from relay's perspective)

    async def close_rtc_relay_connection(self):
        # Stop stats monitoring
        if self.relay_stats_monitor is not None:
            await self.relay_stats_monitor.stop()
            self.relay_stats_monitor = None
        if self.client_to_relay_stats_monitor is not None:
            await self.client_to_relay_stats_monitor.stop()
            self.client_to_relay_stats_monitor = None
        
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

