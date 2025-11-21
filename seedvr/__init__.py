# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SeedVR: Diffusion-based Video Restoration Framework

SeedVR is a framework for video restoration using diffusion transformers.
Includes two models:
- SeedVR: Multi-step diffusion transformer for generic video restoration
- SeedVR2: One-step video restoration via diffusion adversarial post-training
"""

import os
import sys
from pathlib import Path

__version__ = "1.0.0"
__author__ = "Jianyi Wang, Bytedance Ltd."
__license__ = "Apache-2.0"
__copyright__ = "Copyright (c) 2025 Bytedance Ltd. and/or its affiliates"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__copyright__",
]

SEEDVR_LOCATION = Path(os.path.abspath(__file__)).parents[0]

CONFIGS_PATH = SEEDVR_LOCATION / 'configs'

sys.path.insert(0, str(SEEDVR_LOCATION))
