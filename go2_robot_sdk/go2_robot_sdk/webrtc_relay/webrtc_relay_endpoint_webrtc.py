from aiortc import RTCPeerConnection, RTCDataChannel, RTCConfiguration, RTCSessionDescription
from fastapi import Depends, APIRouter
import logging
from pydantic import BaseModel
import typing as t

from go2_robot_sdk.webrtc_relay.webrtc_relay_app_state import get_app_state, WebRTCRelayAppState
from go2_robot_sdk.webrtc_relay.webrtc_relay_exceptions import StateException

logger = logging.getLogger(__name__)
router = APIRouter()

class OfferArgs(BaseModel):
    sdp: str
    type: str

class OfferReply(BaseModel):
    sdp: str
    type: str


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

        # Attach GO2 video (if present)
        if state.go2_video_track:
            logger.info(f"adding go2 video track to new relay connection")
            new_relay_peer_connection.addTrack(state.media_relay.subscribe(state.go2_video_track))

        # SDP handshake
        logger.info(f"relay RTC setting remote description")
        await new_relay_peer_connection.setRemoteDescription(RTCSessionDescription(sdp=sdp.sdp, type=sdp.type))
        logger.info(f"relay RTC creating answer")
        answer = await new_relay_peer_connection.createAnswer()
        
        logger.info(f"relay RTC setting local description")
        await new_relay_peer_connection.setLocalDescription(answer)

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
