# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""CPU coverage for grouping NCCL M2N weights into fused reshard calls."""

import pytest
import torch

from verl.checkpoint_engine.nccl_m2n_checkpoint_engine import NCCLM2NLocalWeight, _fusion_groups, _stack_weight
from verl.checkpoint_engine.reshard_layout import build_reshard_layouts
from verl.models.transformers.hf_dense_decoder_tp import infer_dense_decoder_tp_shard_dim

SOURCE_SHARDS = 4
# A small Qwen3-shaped decoder layer: hidden 64, 4 query heads and 2 KV heads of 16, MLP 128.
LAYER = {
    "self_attn.q_proj.weight": (64, 64),
    "self_attn.k_proj.weight": (32, 64),
    "self_attn.v_proj.weight": (32, 64),
    "self_attn.o_proj.weight": (64, 64),
    "self_attn.q_norm.weight": (16,),
    "self_attn.k_norm.weight": (16,),
    "mlp.gate_proj.weight": (128, 64),
    "mlp.up_proj.weight": (128, 64),
    "mlp.down_proj.weight": (64, 128),
    "input_layernorm.weight": (64,),
    "post_attention_layernorm.weight": (64,),
}


def _weight(name: str, global_shape: tuple[int, ...]) -> NCCLM2NLocalWeight:
    """One source rank's FSDP shard (dim 0 over four ranks) with the HF dense-decoder TP placement."""

    local = list(global_shape)
    local[0] //= SOURCE_SHARDS
    return NCCLM2NLocalWeight(
        name=name,
        tensor=torch.zeros(local, dtype=torch.bfloat16),
        global_shape=torch.Size(global_shape),
        destination_shard_dim=infer_dense_decoder_tp_shard_dim(name),
        source_shard_dim=0,
        source_shard_size=SOURCE_SHARDS,
        source_mesh_dims=(1, SOURCE_SHARDS),
    )


def _decoder(layers: int) -> list[NCCLM2NLocalWeight]:
    """Weights in FSDP export order, with a tied head exported under both names."""

    weights = [_weight("model.embed_tokens.weight", (256, 64))]
    for layer in range(layers):
        weights += [_weight(f"model.layers.{layer}.{suffix}", shape) for suffix, shape in LAYER.items()]
    return weights + [_weight("model.norm.weight", (64,)), _weight("lm_head.weight", (256, 64))]


def _nbytes(weight: NCCLM2NLocalWeight) -> int:
    return weight.tensor.numel() * weight.tensor.element_size()


def test_unbounded_cap_forms_one_group_per_layout_family():
    groups = [[w.name for w in group] for group in _fusion_groups(_decoder(3), cap_bytes=1 << 30)]

    def layer_names(*suffixes):
        return [f"model.layers.{layer}.{s}" for layer in range(3) for s in suffixes]

    assert groups == [
        ["model.embed_tokens.weight", "lm_head.weight"],
        layer_names("self_attn.q_proj.weight"),
        layer_names("self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        layer_names("self_attn.o_proj.weight"),
        layer_names("self_attn.q_norm.weight", "self_attn.k_norm.weight"),
        layer_names("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        layer_names("mlp.down_proj.weight"),
        layer_names("input_layernorm.weight", "post_attention_layernorm.weight") + ["model.norm.weight"],
    ]


@pytest.mark.parametrize("cap_bytes", [1, 2048, 8192, 20000, 1 << 30])
def test_cap_bounds_the_tensors_held_together(cap_bytes):
    weights = _decoder(4)
    index = {w.name: i for i, w in enumerate(weights)}
    groups = list(_fusion_groups(weights, cap_bytes))

    # Every weight is sent exactly once, members keep their export order.
    names = [w.name for group in groups for w in group]
    assert sorted(names) == sorted(w.name for w in weights)
    for group in groups:
        positions = [index[w.name] for w in group]
        assert positions == sorted(positions)
        # All weights between a group's first and last member were held at once.
        if len(group) > 1:
            assert sum(_nbytes(w) for w in weights[positions[0] : positions[-1] + 1]) <= cap_bytes


def test_weights_without_room_for_a_stack_dimension_travel_alone():
    weights = [_weight(f"model.layers.{i}.mlp.up_proj.weight", (8, 4)) for i in range(2)]
    three_d = [
        NCCLM2NLocalWeight(
            name=f"model.layers.{i}.mlp.experts.w",
            tensor=torch.zeros(2, 4, 4),
            global_shape=torch.Size((8, 4, 4)),
            destination_shard_dim=0,
            source_shard_dim=0,
            source_shard_size=SOURCE_SHARDS,
        )
        for i in range(2)
    ]
    groups = [[w.name for w in group] for group in _fusion_groups(weights + three_d, cap_bytes=1 << 30)]
    assert groups == [[w.name for w in weights], [three_d[0].name], [three_d[1].name]]


@pytest.mark.parametrize("name", ["self_attn.k_proj.weight", "self_attn.o_proj.weight", "input_layernorm.weight"])
@pytest.mark.parametrize("destination", [(2, 2), (1, 4), (4, 1)])
def test_stacked_layout_keeps_every_member_placement(name, destination):
    member = _weight(f"model.layers.0.{name}", LAYER[name])
    count = 5
    stacked = _stack_weight(member, torch.zeros(count, *member.tensor.shape, dtype=member.tensor.dtype), count)
    sizes = dict(
        source_replica_size=1,
        source_shard_size=SOURCE_SHARDS,
        destination_replica_size=destination[0],
        destination_shard_size=destination[1],
    )
    member_layouts = build_reshard_layouts(member, **sizes)
    stacked_layouts = build_reshard_layouts(stacked, **sizes)

    for alone, together in zip(member_layouts, stacked_layouts, strict=True):
        assert together.mesh_dims == alone.mesh_dims
        assert together.start_rank == alone.start_rank
        assert together.local_shape == torch.Size((count, *alone.local_shape))
        assert together.placements == tuple(None if p is None else p + 1 for p in alone.placements)

    # Cutting the stacked global tensor gives every rank the stack of the
    # pieces it would have received weight by weight.
    full = [torch.randn(member.global_shape) for _ in range(count)]
    dim = member.destination_shard_dim
    shards = destination[1]
    for rank in range(shards):
        alone = [f if dim is None else f.chunk(shards, dim=dim)[rank] for f in full]
        together = torch.stack(full)
        together = together if dim is None else together.chunk(shards, dim=dim + 1)[rank]
        assert torch.equal(together, torch.stack(alone))
