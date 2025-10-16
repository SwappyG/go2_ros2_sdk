# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
Full Go2 WebRTC connection implementation with clean architecture.
Handles WebRTC peer connection and data channel communication with Go2 robot.
Originally forked from https://github.com/tfoldi/go2-webrtc and 
https://github.com/legion1581/go2_webrtc_connect
Big thanks to @tfoldi (Földi Tamás) and @legion1581 (The RoboVerse Discord Group)
"""

import json
import logging
import base64
from typing import Callable, Any, Coroutine, TypeAlias
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

from go2_robot_sdk.infrastructure.webrtc.crypto.encryption import CryptoUtils, ValidationCrypto, PathCalculator, EncryptionError
from go2_robot_sdk.infrastructure.webrtc.http_client import HttpClient, WebRTCHttpError
from go2_robot_sdk.infrastructure.webrtc.data_decoder import (
    WebRTCDataDecoder,
    DataDecodingError, # pyright: ignore[reportUnusedImport]
)
from go2_robot_sdk.infrastructure.webrtc.data_decoder import deal_array_buffer as legacy_deal_array_buffer
from go2_robot_sdk.domain.entities.robot_data import RobotData
import go2_robot_sdk.infrastructure.webrtc.go2_message_parsers as go2_parsers

logger = logging.getLogger(__name__)

OnValidatedCB: TypeAlias = Callable[[str], None]
OnMessageCB: TypeAlias = Callable[[RobotData], None]
OnOpenCB: TypeAlias = Callable[[], None]
OnVideoFrameCB: TypeAlias = Callable[[MediaStreamTrack, str], Coroutine[None, None, None]]


class Go2ConnectionError(Exception):
    """Custom exception for Go2 connection errors"""


class Go2Connection:
    """Full WebRTC connection to Go2 robot with encryption and proper signaling"""
    
    hex_to_base64 = ValidationCrypto.hex_to_base64
    encrypt_key = ValidationCrypto.encrypt_key
    encrypt_by_md5 = ValidationCrypto.encrypt_by_md5
    deal_array_buffer = staticmethod(legacy_deal_array_buffer)

    __enter__ = lambda self: self
    __exit__ = lambda self, type_, value, traceback: self.shutdown()

    def __init__(
        self,
        robot_ip: str,
        robot_num: int,
        token: str = "",
        on_validated: OnValidatedCB | None = None,
        on_message: OnMessageCB | None = None,
        on_open: OnOpenCB | None = None,
        on_video_frame: OnVideoFrameCB | None = None,
        decode_lidar: bool = True,
        decode_message: bool = True,
    ):
        self.pc = RTCPeerConnection()
        self.robot_ip = robot_ip
        self.robot_num = str(robot_num)
        self.token = token
        self.is_validated = False
        
        # Callbacks
        self.on_validated = on_validated
        self.on_message = on_message
        self.on_open = on_open
        self.on_video_frame = on_video_frame
        self.decode_lidar = decode_lidar
        self._decode_message = decode_message
        
        # Initialize components
        self.http_client = HttpClient(timeout=10.0)

        self.data_decoder = WebRTCDataDecoder(enable_lidar_decoding=decode_lidar)
        
        # Setup data channel
        self.data_channel = self.pc.createDataChannel("data") # id=0
        self.data_channel.on("open", self.on_data_channel_open)
        self.data_channel.on("message", self.on_data_channel_message)
        
        # Setup peer connection events
        self.pc.on("track", self.on_track)
        self.pc.on("connectionstatechange", self.on_connection_state_change)
        
        # Add video transceiver if video callback provided
        if self.on_video_frame:
            self.pc.addTransceiver("video", direction="recvonly")
    
    def on_connection_state_change(self) -> None:
        """Handle peer connection state changes"""
        logger.info(f"Connection state is {self.pc.connectionState}")
        
        # Note: Validation is handled after successful WebRTC connection
        # in the original implementation, not here
    
    def on_data_channel_open(self, *_args) -> None:
        """Handle data channel open event"""
        logger.info("Data channel is open")
        # Force data channel to open state if needed (workaround)
        if self.data_channel.readyState != "open":
            self.data_channel._setReadyState("open")  # pyright: ignore[reportPrivateUsage]
        
        if self.on_open:
            self.on_open()

    def on_data_channel_message(self, message: Any) -> None:
        """Handle incoming data channel messages"""
        try:
            # if we got a message, we're definely connected. For some reason, the "open" hook
            # doesn't always fire. Let's force it here. 
            if self.data_channel.readyState != "open":
                self.data_channel._setReadyState("open") # pyright: ignore[reportPrivateUsage]

            logger.debug(f"Received message: {message}")

            # if we're not validated, we have to parse every message we get to see if there's
            # a validation message
            raw_message_obj = None
            if not self.is_validated and isinstance(message, str):
                raw_message_obj = go2_parsers.parse_datachannel_message(message)
                if raw_message_obj['type'] == 'validation':
                    self.validate_robot_conn(raw_message_obj['data'])
                    return

            # if there's no callback, don't bother parsing
            # NOTE - in the future, we may still want to parse to display error messages
            if self.on_message is None:
                return

             
            # If we're not supposed to decode the message (ie, something upstream will handle it),
            # then just forward what we got
            if not self._decode_message:
                logger.debug(f"decode is set to false, sending raw message back")
                self.on_message(RobotData(robot_id=self.robot_num, timestamp=0.0, raw_message=message))
                return
                
            logger.debug(f"decode is set to true, decoding message")
            robot_data = None
            if isinstance(message, str):
                # we may have already parsed the data during the validation step. If not tho, parse 
                # it now
                if raw_message_obj is None:
                    raw_message_obj = go2_parsers.parse_datachannel_message(message)
                robot_data = go2_parsers.process_webrtc_message(raw_message_obj, self.robot_num)
            
            elif isinstance(message, bytes):
                lidar_frame = legacy_deal_array_buffer(message, perform_decode=self.decode_lidar)
                logger.info(f"{lidar_frame=}")
                if lidar_frame is not None:
                    robot_data = go2_parsers.process_webrtc_message(lidar_frame, self.robot_num)
                    
            else: 
                logger.warning(f"unknown message type receieved from on_message callback. {message=}")
                return
            
            if robot_data is not None:
                robot_data.raw_message = message
                self.on_message(robot_data)
                
        except Exception as exception:
            logger.error(f"Error processing data channel message: {exception=}")
    
    async def on_track(self, track: MediaStreamTrack) -> None:
        """Handle incoming media tracks (video)"""
        if track.kind != "video":
            logger.info(f"Received a track, but it wasn't video: {track=}")
            return None
        
        logger.info(f"Received a video track: {track=}")
        if not self.on_video_frame:
            logger.warning(f"there's no callback registered to consume video track")
            return
        
        try:
            await self.on_video_frame(track, self.robot_num)
        except Exception as exception:
            logger.error(f"Error in video frame callback: {exception=}")
    
    def validate_robot_conn(self, message: str) -> None:
        """Handle robot validation response"""
        logger.info(f"received validation data: {message=}")
        try:
            if message == "Validation Ok.":
                logger.info("Robot validation successful. Setting video to ON")
                # Turn on video
                self.publish("", "on", "vid")
                
                self.is_validated = True
                
                if self.on_validated:
                    logger.info("validation hook registered, calling hook")
                    self.on_validated(self.robot_num)
                    
            else:
                logger.info("got validation key, encrypting and sending back")
                # Send encrypted validation response
                validation_key = message
                encrypted_key = ValidationCrypto.encrypt_key(validation_key)
                self.publish("", encrypted_key, "validation")
                
        except Exception as e:
            logger.error(f"Error in robot validation: {e}")
    
    def publish(self, topic: str, data: Any, msg_type: str = "msg") -> None:
        """
        Publish message to data channel.
        
        Args:
            topic: Message topic
            data: Message data  
            msg_type: Message type
        """
        try:
            payload = {
                "type": msg_type,
                "topic": topic,
                "data": data
            }
            print(payload)
            payload_str = json.dumps(payload)
            print(f"-> Sending message {payload_str}")
            self.data_channel.send(payload_str)
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e=}")
    
    def publish_json_str(self, json_str: str) -> None:
        try:
            if self.data_channel.readyState != "open":
                logger.warning(f"Data channel is not open. State is {self.data_channel.readyState}")
                return
            
            print(f"-> Sending message {json_str}")
            self.data_channel.send(json_str)
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")

    async def disableTrafficSaving(self, switch: bool) -> bool:
        """
        Disable traffic saving mode for better data transmission.
        Should be turned on when subscribed to ulidar topic.
        
        Args:
            switch: True to disable traffic saving, False to enable
            
        Returns:
            True if successful
        """
        try:
            data = {
                "req_type": "disable_traffic_saving",
                "instruction": "on" if switch else "off"
            }
            
            self.publish("", data, "rtc_inner_req")
            logger.info(f"DisableTrafficSaving: {data['instruction']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set traffic saving: {e}")
            return False
    
    async def connect(self) -> None:
        """Establish WebRTC connection to robot with full encryption"""
        try:
            logger.info("Trying to send SDP using full encrypted method...")
            
            # Step 1: Create WebRTC offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            sdp_offer = self.pc.localDescription
            sdp_offer_json = {
                "id": "STA_localNetwork",
                "sdp": sdp_offer.sdp,
                "type": sdp_offer.type,
                "token": self.token
            }
            
            new_sdp = json.dumps(sdp_offer_json)
            
            # Step 2: Get robot's public key
            try:
                response = self.http_client.get_robot_public_key(self.robot_ip)
                if not response:
                    raise Go2ConnectionError("Failed to get public key response")
                
                # Decode the response text from base64
                decoded_response = base64.b64decode(response.text).decode('utf-8')
                decoded_json = json.loads(decoded_response)
                
                # Extract the 'data1' field from the JSON
                data1 = decoded_json.get('data1')
                if not data1:
                    raise Go2ConnectionError("No data1 field in public key response")
                
                # Extract the public key from 'data1'
                public_key_pem = data1[10:len(data1)-10]
                path_ending = PathCalculator.calc_local_path_ending(data1)
                
                logger.info(f"Extracted path ending: {path_ending}")
                
            except (WebRTCHttpError, EncryptionError) as e:
                raise Go2ConnectionError(f"Failed to get robot public key: {e}")
            
            # Step 3: Encrypt and send SDP
            try:
                # Generate AES key
                aes_key = CryptoUtils.generate_aes_key()
                
                # Load Public Key
                public_key = CryptoUtils.rsa_load_public_key(public_key_pem)
                
                # Encrypt the SDP and AES key
                encrypted_body = {
                    "data1": CryptoUtils.aes_encrypt(new_sdp, aes_key),
                    "data2": CryptoUtils.rsa_encrypt(aes_key, public_key),
                }
                
                # Send the encrypted data
                response = self.http_client.send_encrypted_sdp(
                    self.robot_ip, path_ending, encrypted_body
                )
                
                if not response:
                    raise Go2ConnectionError("Failed to send encrypted SDP")
                
                # Decrypt the response
                decrypted_response = CryptoUtils.aes_decrypt(response.text, aes_key)
                peer_answer = json.loads(decrypted_response)
                
                # Set remote description
                answer = RTCSessionDescription(
                    sdp=peer_answer['sdp'], 
                    type=peer_answer['type']
                )
                await self.pc.setRemoteDescription(answer)
                
                logger.info(f"Successfully established WebRTC connection to robot {self.robot_num}")
                
            except (WebRTCHttpError, EncryptionError) as e:
                raise Go2ConnectionError(f"Failed to complete encrypted handshake: {e}")
            
        except Go2ConnectionError:
            raise
        except Exception as e:
            raise Go2ConnectionError(f"Unexpected error during connection: {e}")
    
    async def disconnect(self) -> None:
        """Close WebRTC connection and cleanup resources"""
        try:
            # Close peer connection
            await self.pc.close()
        except Exception as e:
            logger.error(f"error closing peer connection: {e=}")
            
        try:
            self.data_channel.close()
        except Exception as e:
            logger.error(f"error closing peer connection datachannel: {e=}")
        
        try:
            self.http_client.close()
        except Exception as e:
            logger.error(f"error closing http client: {e=}")
            
        logger.info(f"Disconnected from robot {self.robot_num}")
            
    
