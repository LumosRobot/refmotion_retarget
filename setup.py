# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import find_packages, setup

setup(
    name="refmotion_retarget",
    version="0.1.0",
    packages=find_packages(),
    description="Retarget humanoid reference motions",
    python_requires=">=3.10",
    install_requires=[
        "mujoco==3.3.7",
        "mujoco-python-viewer",
        "numpy",
        "pytorch_lightning",
        "numpy-stl",
        "vtk",
        "patchelf",
        "termcolor",
        "torchgeometry",
        "numpy==1.23.5",
        "scipy",
        "joblib>=1.2.0",
        "opencv-python==4.6.0.66",
        "tqdm ",
        "pyyaml",
        "scikit-image",
        "gym",
        "lxml",
        "human_body_prior",
        "autograd",
        "scikit-learn",
        "chumpy",
        "patchelf",
        "pyvirtualdisplay",
        "chumpy",
        "patchelf",
        "lxml",
        "chardet",
        "cchardet",
        "imageio-ffmpeg",
        "easydict",
        "open3d",
        "gdown",
        "scikit-image",
        "glfw",
        "hydra-core",
        "loop_rate_limiters",
    "mink",
    "qpsolvers[proxqp]",
    "rich",
    "tqdm",
    "opencv-python",
    "natsort",
    "psutil",
    "smplx @ git+https://github.com/vchoutas/smplx",
    "protobuf",
    "redis[hiredis]",
    "imageio[ffmpeg]",
    ],
)
