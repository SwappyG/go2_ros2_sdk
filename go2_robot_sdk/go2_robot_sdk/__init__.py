# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import sys
import os
import logging
from pathlib import Path

import go2_robot_sdk

# When used as pure Python ( Poetry/pip ), package is not in ament index
_pkg_root = Path(go2_robot_sdk.__file__).parent.parent
try:
    from ament_index_python.packages import PackageNotFoundError  # pyright: ignore[reportMissingImports]
    from ament_index_python import get_package_share_directory  # pyright: ignore[reportMissingImports]
    try:
        libs_path = os.path.join(get_package_share_directory('go2_robot_sdk'), 'external_lib')
    except PackageNotFoundError:
        libs_path = str(_pkg_root / 'external_lib')
except ImportError:
    libs_path = str(_pkg_root / 'external_lib')

logger = logging.getLogger(__name__)

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
