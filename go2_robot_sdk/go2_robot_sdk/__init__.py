# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import sys
import os
import logging
from pathlib import Path

import go2_robot_sdk

try:
    from ament_index_python import get_package_share_directory  # pyright: ignore[reportMissingImports]
except ImportError:
    get_package_share_directory = lambda _: str(Path(go2_robot_sdk.__file__).parent)

logger = logging.getLogger(__name__)

libs_path = os.path.join(
    get_package_share_directory('go2_robot_sdk'),
    'external_lib'
)

try:
    import aioice  # pyright: ignore[reportUnusedImport]
except ImportError:
    if os.path.exists(os.path.join(libs_path, 'aioice', '__init__.py')):
        sys.path.insert(0, os.path.join(libs_path, 'aioice'))
        sys.path.insert(0, os.path.join(libs_path))

        logger.info('Patched aioice added to sys.path: {}'.format(sys.path))
    else:
        logger.error("aioice submodule is not initalized. please init submodules recursively")
        exit(-1)
