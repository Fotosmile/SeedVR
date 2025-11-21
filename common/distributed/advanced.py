# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Advanced distributed functions for sequence parallel (non-distributed implementation).
"""

from typing import Optional, List
import torch

from .basic import get_global_rank, get_world_size


def get_data_parallel_group() -> Optional[None]:
    """
    Get data parallel process group.
    """
    return None


def get_sequence_parallel_group() -> Optional[None]:
    """
    Get sequence parallel process group.
    """
    return None


def get_sequence_parallel_cpu_group() -> Optional[None]:
    """
    Get sequence parallel CPU process group.
    """
    return None


def get_data_parallel_rank() -> int:
    """
    Get data parallel rank.
    """
    return 0


def get_data_parallel_world_size() -> int:
    """
    Get data parallel world size.
    """
    return 1


def get_sequence_parallel_rank() -> int:
    """
    Get sequence parallel rank.
    """
    return 0


def get_sequence_parallel_world_size() -> int:
    """
    Get sequence parallel world size.
    """
    return 1


def get_model_shard_cpu_intra_group() -> Optional[None]:
    """
    Get the CPU intra process group of model sharding.
    """
    return None


def get_model_shard_cpu_inter_group() -> Optional[None]:
    """
    Get the CPU inter process group of model sharding.
    """
    return None


def get_model_shard_intra_group() -> Optional[None]:
    """
    Get the GPU intra process group of model sharding.
    """
    return None


def get_model_shard_inter_group() -> Optional[None]:
    """
    Get the GPU inter process group of model sharding.
    """
    return None


def init_sequence_parallel(sequence_parallel_size: int):
    """
    Initialize sequence parallel (no-op in non-distributed mode).
    """
    pass


def init_model_shard_group(*, sharding_strategy=None, device_mesh: Optional[None] = None):
    """
    Initialize process group of model sharding (no-op in non-distributed mode).
    """
    pass


def get_sequence_parallel_global_ranks() -> List[int]:
    """
    Get all global ranks of the sequence parallel process group
    that the caller rank belongs to.
    """
    return [0]


def get_next_sequence_parallel_rank() -> int:
    """
    Get the next global rank of the sequence parallel process group
    that the caller rank belongs to.
    """
    return 0


def get_prev_sequence_parallel_rank() -> int:
    """
    Get the previous global rank of the sequence parallel process group
    that the caller rank belongs to.
    """
    return 0