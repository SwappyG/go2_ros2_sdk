# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause


import os
from glob import glob
from setuptools import setup
from setuptools import find_packages
import pathlib
import tomllib

package_name = 'go2_robot_sdk'

# Load metadata from pyproject.toml and convert to plain Python literals
pyproject_path = pathlib.Path(__file__).parent / 'pyproject.toml'
project_metadata = {}
if pyproject_path.exists():
    with pyproject_path.open('rb') as f:
        data = tomllib.load(f)
        project_metadata = data.get('project', {})

def _console_scripts_from_pyproject(scripts_table: dict) -> list:
    if not scripts_table:
        return []
    return [f"{name} = {ref}" for name, ref in scripts_table.items()]

setup(
    name=project_metadata.get('name', package_name),
    version=project_metadata.get('version', '0.0.0'),
    description=project_metadata.get('description', ''),
    long_description=(pathlib.Path(project_metadata.get('readme', 'README.md')).read_text() if project_metadata.get('readme') else ''),
    python_requires=project_metadata.get('requires-python'),
    install_requires=project_metadata.get('dependencies', []),
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*'))),
        (os.path.join('share', package_name, 'dae'), glob(os.path.join('dae', '*'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'calibration'), glob(os.path.join('calibration', '*'))),
        (os.path.join('share', package_name, 'external_lib'), ['external_lib/libvoxel.wasm']),
        (os.path.join('share', package_name, 'external_lib/aioice'), glob(os.path.join('external_lib/aioice/src/aioice', '*'))),


    ],
    entry_points={
        'console_scripts': _console_scripts_from_pyproject(project_metadata.get('scripts'))
    },
    zip_safe=True,
)
