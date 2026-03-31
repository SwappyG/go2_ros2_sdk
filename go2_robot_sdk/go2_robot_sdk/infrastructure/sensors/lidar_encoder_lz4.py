from __future__ import annotations

from typing import Any

import lz4.block
import numpy as np
import numpy.typing as npt


def compress(decompressed_data: bytes) -> bytes:
    """
    LZ4 block compress without a stored uncompressed-size prefix, so it matches
    ``lz4.block.decompress(..., uncompressed_size=decomp_size)`` in the decoder.
    """
    return lz4.block.compress(decompressed_data, store_size=False)


def points_to_bits(
    points: npt.NDArray[np.floating[Any]],
    origin: tuple[float, float, float] | list[float] | npt.NDArray[np.floating[Any]],
    resolution: float,
) -> tuple[bytes, int]:
    """
    Pack voxel grid coordinates into the uncompressed buffer layout expected by
    `bits_to_points` in `lidar_decoder_lz4.py`.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts = (pts - np.asarray(origin, dtype=np.float64)) / float(resolution)
    xi = np.round(pts).astype(np.int64)
    if xi.size == 0:
        return bytes(0), 0

    x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]

    # This part converts the points into the buffer layout expected by
    n_slice = y * 0x10 + (x // 8)
    n = z * 0x800 + n_slice
    masks = (1 << (7 - (x % 8))).astype(np.uint8)

    # This part filters the points to only include points in the valid range
    valid = (
        (x >= 0)
        & (x < 128)
        & (y >= 0)
        & (y < 128)
        & (z >= 0)
        & (n_slice >= 0)
        & (n_slice < 0x800)
    )

    n_all = n[valid]
    masks_all = masks[valid]
    if n_all.size == 0:
        return bytes(0), 0
    inferred_size = int(np.max(n_all)) + 1
    in_range = n_all < inferred_size
    n_v = n_all[in_range]
    masks_v = masks_all[in_range]

    # This part encodes the points into the buffer
    buf = np.zeros(inferred_size, dtype=np.uint8)
    np.bitwise_or.at(buf, n_v, masks_v)

    return buf.tobytes(), inferred_size


class LidarEncoderLz4:
    def encode(self, points: npt.NDArray[np.floating[Any]], data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            points: (N, 3) float array in world coordinates.
            data: Must include ``origin`` and ``resolution``.
                Optional ``src_size`` can be provided to force/pad output size.

        Returns:
            Dict with key ``compressed`` holding LZ4-compressed bytes.
        """
        decompressed, inferred_src_size = points_to_bits(
            points,
            data["origin"],
            float(data["resolution"])
        )
        compressed = compress(decompressed)
        return {"compressed": compressed, "src_size": inferred_src_size}


LidarEncoder = LidarEncoderLz4
