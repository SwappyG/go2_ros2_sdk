import typing as t
import numpy.typing as npt
import numpy as np

class DecodeResult(t.TypedDict):
    point_count: int
    face_count: int
    positions: npt.NDArray[np.uint8]
    uvs: npt.NDArray[np.uint8]
    indices: npt.NDArray[np.uint32]
    