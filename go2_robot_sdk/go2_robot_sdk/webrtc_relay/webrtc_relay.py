import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import typing as t # pyright: ignore[reportUnusedImport]
from go2_robot_sdk.webrtc_relay.webrtc_relay_app_state import WebRTCRelayAppState
from go2_robot_sdk.webrtc_relay.webrtc_relay_endpoint_go2 import router as go2_router
from go2_robot_sdk.webrtc_relay.webrtc_relay_endpoint_webrtc import router as webrtc_router
from go2_robot_sdk.webrtc_relay.webrtc_relay_exceptions import StateException


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    logger.info("starting fastapi")
    fastapi_app.state.state = WebRTCRelayAppState()
    # clean shutdown
    try:
        logger.info("yielding fastapi app")
        yield
    finally:
        logger.info("cleaning up fastapi")
        if fastapi_app.state.state.go2:
            await fastapi_app.state.state.go2.disconnect()

        # Close PC connection if present
        if fastapi_app.state.state.relay_rtc_peer_connection:
            await fastapi_app.state.state.relay_rtc_peer_connection.close()
            fastapi_app.state.state.relay_rtc_peer_connection = None
            fastapi_app.state.state.relay_rtc_data_channel = None
        # Close Go2 connection if present
        if fastapi_app.state.state.go2:
            await fastapi_app.state.state.go2.disconnect()
            fastapi_app.state.state.go2 = None
            fastapi_app.state.state.go2_video_track = None


app = FastAPI(lifespan=lifespan)
app.include_router(go2_router, prefix="/go2")
app.include_router(webrtc_router, prefix="/webrtc")


@app.exception_handler(StateException)
def _app_state_exception_handler(request: Request, exc: StateException):  # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=409, content={"detail": str(exc), "exception_type": "state_exception"})

@app.exception_handler(ValueError)
def _app_value_error_handler(request: Request, exc: ValueError): # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=422, content={"detail": str(exc), "exception_type": "value_error"})

@app.exception_handler(KeyError)
def _app_key_error_handler(request: Request, exc: KeyError): # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=422, content={"detail": str(exc), "exception_type": "key_error"})

@app.exception_handler(IndexError)
def _app_index_error_handler(request: Request, exc: IndexError): # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=422, content={"detail": str(exc), "exception_type": "index_error"})

@app.exception_handler(RuntimeError)
def _app_runtime_error_handler(request: Request, exc: RuntimeError): # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=500, content={"detail": str(exc), "exception_type": "runtime_error"})

@app.exception_handler(TimeoutError)
def _app_timeout_error_handler(request: Request, exc: TimeoutError): # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=504, content={"detail": str(exc), "exception_type": "timeout_error"})

@app.exception_handler(asyncio.TimeoutError)
def _app_asyncio_timeout_error_handler(request: Request, exc: asyncio.TimeoutError): # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=504, content={"detail": str(exc), "exception_type": "asyncio_timeout_error"})

@app.exception_handler(Exception)
def _app_unhandled_error_handler(request: Request, exc: Exception): # pyright: ignore[reportUnusedFunction]
    return JSONResponse(status_code=500, content={"detail": str(exc), "exception_type": type(exc).__name__})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("go2_robot_sdk.webrtc_relay.webrtc_relay:app", host="localhost", port=8000, reload=True, log_level='info')