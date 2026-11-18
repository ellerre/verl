# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests of the TorchStore checkpoint engine that start no TorchStore.

The last two need the ``torchstore`` package (its planning code only) and are
skipped without it.
"""

import asyncio
import socket

import pytest
import torch
from torch.distributed.tensor import DTensor, Shard

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.checkpoint_engine.torchstore_checkpoint_engine import (
    TorchStoreCheckpointEngine,
    TorchStoreMasterMetadata,
    TorchStoreWorkerInfo,
    _virtual_mesh,
    assign_spmd_ranks,
    destination_layout,
    destination_tensors,
)

MESH = {"source_dp": 1, "source_shard_size": 4, "destination_dp": 2, "destination_shard_size": 2}

# One of each placement of a dense decoder on a tensor-parallel replica; every
# sharded dimension divides by the 4 trainer ranks and the 2 rollout workers.
MANIFEST = [
    ("model.embed_tokens.weight", (64, 16), "bfloat16"),
    ("model.layers.0.self_attn.q_proj.weight", (32, 16), "bfloat16"),
    ("model.layers.0.self_attn.o_proj.weight", (16, 32), "bfloat16"),
    ("model.layers.0.mlp.down_proj.weight", (16, 48), "bfloat16"),
    ("model.layers.0.input_layernorm.weight", (16,), "bfloat16"),
]


def _engine(**kwargs):
    return TorchStoreCheckpointEngine(bucket_size=1 << 20, **{**MESH, **kwargs})


def _master(**kwargs):
    fields = {
        "host": "10.0.0.1",
        "store_port": 40000,
        "monarch_port": 41000,
        **MESH,
        "mode": "store",
        "strategy": "local_rank",
        "routing": True,
    }
    return TorchStoreMasterMetadata(**{**fields, **kwargs})


def _infos(trainer_hosts, rollout_hosts, master=None):
    master = master or _master()
    trainers = [TorchStoreWorkerInfo(host, master if index == 0 else None) for index, host in enumerate(trainer_hosts)]
    return trainers + [TorchStoreWorkerInfo(host) for host in rollout_hosts]


def _destination(shard_rank, shard_size=2):
    weights, nbytes = destination_layout(MANIFEST, shard_size)
    buffer = torch.zeros(nbytes, dtype=torch.uint8)
    destination, outputs = destination_tensors(weights, buffer, shard_size, shard_rank)
    return weights, buffer, destination, outputs


def test_backend_is_registered():
    assert CheckpointEngineRegistry.get("torchstore") is TorchStoreCheckpointEngine
    assert TorchStoreCheckpointEngine.wire_format == "rank_local_named_tensors"
    assert TorchStoreCheckpointEngine.is_master_rank(0) and not TorchStoreCheckpointEngine.is_master_rank(1)


def test_spmd_ranks_trainer_node_then_rollout_node():
    trainers, rollouts, local_world_size = assign_spmd_ranks(["a"] * 4, ["b"] * 4)
    assert trainers == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert rollouts == [(4, 0), (5, 1), (6, 2), (7, 3)]
    assert local_world_size == 4


def test_spmd_ranks_on_one_host():
    trainers, rollouts, local_world_size = assign_spmd_ranks(["a", "a"], ["a", "a"])
    assert trainers == [(0, 0), (1, 1)]
    assert rollouts == [(2, 2), (3, 3)]
    assert local_world_size == 4


def test_spmd_ranks_trainer_over_two_hosts():
    trainers, rollouts, local_world_size = assign_spmd_ranks(["a", "a", "b", "b"], ["c", "c", "d", "d"])
    assert trainers == [(0, 0), (1, 1), (2, 0), (3, 1)]
    assert rollouts == [(4, 0), (5, 1), (6, 0), (7, 1)]
    assert local_world_size == 2


def test_spmd_ranks_rollout_workers_interleaved_over_hosts():
    _, rollouts, _ = assign_spmd_ranks(["a", "a"], ["b", "c", "b", "c"])
    assert rollouts == [(2, 0), (4, 0), (3, 1), (5, 1)]


def test_spmd_ranks_reject_uneven_hosts():
    with pytest.raises(ValueError, match="same number of workers"):
        assign_spmd_ranks(["a"] * 4, ["b"] * 2)


def test_spmd_ranks_reject_interleaved_trainers():
    with pytest.raises(ValueError, match="rank order"):
        assign_spmd_ranks(["a", "b", "a", "b"], ["c", "c", "d", "d"])


def test_destination_layout_shards_by_weight_name():
    manifest = [
        ("model.layers.0.self_attn.q_proj.weight", (8, 6), "bfloat16"),
        ("model.layers.0.self_attn.o_proj.weight", (6, 8), "bfloat16"),
        ("model.norm.weight", (6,), "bfloat16"),
    ]
    weights, nbytes = destination_layout(manifest, 2)
    assert [weight.shard_dim for weight in weights] == [0, 1, None]
    assert [weight.local_shape for weight in weights] == [(4, 6), (6, 4), (6,)]
    assert [weight.nbytes for weight in weights] == [48, 48, 12]
    # Every weight starts on a 64-byte boundary and none overlaps the next.
    assert [weight.offset for weight in weights] == [0, 64, 128]
    assert nbytes == 192
    assert all(weight.dtype is torch.bfloat16 for weight in weights)


def test_destination_layout_unsharded_replica_keeps_full_tensors():
    weights, _ = destination_layout([("anything.bias", (5, 3), "float32")], 1)
    assert weights[0].shard_dim is None and weights[0].local_shape == (5, 3)


def test_destination_layout_rejects_what_it_cannot_shard():
    with pytest.raises(ValueError, match="does not divide"):
        destination_layout([("model.layers.0.self_attn.q_proj.weight", (9, 6), "bfloat16")], 2)
    with pytest.raises(NotImplementedError):
        destination_layout([("anything.bias", (4,), "bfloat16")], 2)


def test_destination_tensors_are_views_of_the_buffer_and_need_no_process_group():
    assert not torch.distributed.is_initialized()
    weights, buffer, destination, outputs = _destination(shard_rank=1)
    assert [name for name, _ in outputs] == [name for name, _, _ in MANIFEST]
    for weight, (_, local) in zip(weights, outputs):
        target = destination[weight.name]
        assert isinstance(target, DTensor) == (weight.shard_dim is not None)
        assert tuple(target.shape) == weight.global_shape
        assert tuple(local.shape) == weight.local_shape and local.dtype is weight.dtype
        assert local.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        if isinstance(target, DTensor):
            assert target.to_local().data_ptr() == local.data_ptr()
    assert not torch.distributed.is_initialized()


def test_configuration_defaults_and_validation():
    assert _engine().routing is True
    assert _engine(strategy="host").routing is False
    assert _engine(mode="direct").routing is False
    assert _engine(routing=False).routing is False
    assert _engine().transfer_dtype is torch.bfloat16
    assert _engine(transfer_dtype=None).transfer_dtype is None
    assert _engine(require_transports=("SharedMemory", "NIXL")).require_transports == ["SharedMemory", "NIXL"]
    assert _engine().store_env == {}
    # An empty value unsets the variable, as None does.
    assert _engine(store_env={"A": 1, "B": "", "C": None}).store_env == {"A": "1", "B": None, "C": None}
    with pytest.raises(ValueError, match="routing requires"):
        _engine(strategy="host", routing=True)
    with pytest.raises(ValueError, match="routing requires"):
        _engine(mode="direct", routing=True)
    with pytest.raises(ValueError, match="mode must be"):
        _engine(mode="mailbox")
    with pytest.raises(ValueError, match="strategy must be"):
        _engine(strategy="global")
    with pytest.raises(ValueError, match="explicit values for destination_dp"):
        TorchStoreCheckpointEngine(bucket_size=1, source_dp=1, source_shard_size=4, destination_shard_size=2)
    with pytest.raises(ValueError, match="positive"):
        _engine(destination_dp=0)


def test_prepare_reports_the_host_and_no_master_record_off_the_master():
    engine = _engine()
    info = engine.prepare()
    assert info == TorchStoreWorkerInfo(socket.gethostname(), None)
    assert engine.prepare() is info


def test_build_topology_assigns_bootstrap_ranks():
    master = _master()
    trainer_kwargs, rollout_kwargs = TorchStoreCheckpointEngine.build_topology(
        4, 4, _infos(["a"] * 4, ["b"] * 4, master)
    )
    assert trainer_kwargs == {
        "rank": [0, 1, 2, 3],
        "local_rank": [0, 1, 2, 3],
        "world_size": [8] * 4,
        "local_world_size": [4] * 4,
        "master_metadata": [master] * 4,
        "role": ["source"] * 4,
        "index": [0, 1, 2, 3],
    }
    assert rollout_kwargs["rank"] == [4, 5, 6, 7]
    assert rollout_kwargs["local_rank"] == [0, 1, 2, 3]
    assert rollout_kwargs["role"] == ["destination"] * 4
    assert rollout_kwargs["index"] == [0, 1, 2, 3]


def test_build_topology_rejects_bad_worker_sets():
    with pytest.raises(ValueError, match="exactly one trainer master"):
        TorchStoreCheckpointEngine.build_topology(
            4, 4, [TorchStoreWorkerInfo("a")] * 4 + [TorchStoreWorkerInfo("b")] * 4
        )
    with pytest.raises(ValueError, match="exactly 4 trainer and 4 rollout"):
        TorchStoreCheckpointEngine.build_topology(4, 2, _infos(["a"] * 4, ["b"] * 2))
    with pytest.raises(ValueError, match="record of every worker"):
        TorchStoreCheckpointEngine.build_topology(4, 4, _infos(["a"] * 4, ["b"] * 4)[:-1] + [None])


def test_init_process_group_rejects_a_mismatch_before_starting_torchstore():
    arguments = {
        "rank": 0,
        "local_rank": 0,
        "world_size": 8,
        "local_world_size": 4,
        "master_metadata": _master(),
        "role": "source",
        "index": 0,
    }
    with pytest.raises(ValueError, match="role must be"):
        _engine().init_process_group(**{**arguments, "role": "relay"})
    with pytest.raises(ValueError, match="world_size=6"):
        _engine().init_process_group(**{**arguments, "world_size": 6})
    with pytest.raises(ValueError, match="does not match local config"):
        _engine(strategy="host").init_process_group(**arguments)
    with pytest.raises(ValueError, match="must keep TorchStore rank"):
        _engine().init_process_group(**{**arguments, "rank": 4})


def test_transfers_need_an_initialized_process_group():
    async def receive(engine):
        return [item async for item in engine.receive_weights()]

    with pytest.raises(RuntimeError, match="not initialized"):
        asyncio.run(_engine().send_weights(iter(())))
    with pytest.raises(RuntimeError, match="not initialized"):
        asyncio.run(receive(_engine()))


def test_torchstore_reads_the_expected_slices_off_the_destination_tensors():
    types = pytest.importorskip("torchstore.transport.types")
    _, _, destination, _ = _destination(shard_rank=1)
    slices = {name: types.Request.from_any(f"policy/{name}", tensor).tensor_slice for name, tensor in destination.items()}
    q_proj = slices["model.layers.0.self_attn.q_proj.weight"]
    assert (q_proj.offsets, q_proj.local_shape, q_proj.global_shape) == ((16, 0), (16, 16), (32, 16))
    down_proj = slices["model.layers.0.mlp.down_proj.weight"]
    assert (down_proj.offsets, down_proj.local_shape, down_proj.global_shape) == ((0, 24), (16, 24), (16, 48))
    # A weight the worker holds whole is a plain tensor: TorchStore reads all of it.
    assert slices["model.layers.0.input_layernorm.weight"] is None


def test_torchstore_routes_trainer_dim0_shards_to_tensor_parallel_slices():
    plan = pytest.importorskip("torchstore.routing.plan")
    utils = pytest.importorskip("torchstore.state_dict_utils")
    # The engine names its storage keys the way TorchStore's layout registration does.
    assert utils._state_dict_storage_key("policy", "lm_head.weight") == "policy/lm_head.weight"

    # Four trainer ranks hold dim-0 shards of every weight, as FSDP2 does.
    publishers = {}
    for rank in range(4):
        mesh = _virtual_mesh(4, rank, "cpu")
        state = {
            name: DTensor.from_local(
                torch.zeros(shape, dtype=torch.float32).chunk(4, dim=0)[rank], mesh, (Shard(0),), run_check=False
            )
            for name, shape, _ in MANIFEST
        }
        slices, dtypes, _ = utils._state_dict_storage_metadata(state, "policy", transfer_dtype=torch.bfloat16)
        publishers[str(rank)] = {key: plan.KeyRegistration(slices[key], dtype=dtypes[key]) for key in slices}

    for shard_rank in range(2):
        _, _, destination, _ = _destination(shard_rank)
        slices, dtypes, _ = utils._state_dict_storage_metadata(destination, "policy")
        registrations = {key: plan.KeyRegistration(slices[key], dtype=dtypes[key]) for key in slices}
        routes = plan.RoutingPlan.for_requester(f"requester/{4 + shard_rank}", registrations, publishers).routes

        def sources(name, routes=routes):
            return sorted(transfer.source for transfer in routes.lookup(f"policy/{name}").transfers)

        # Split on dim 0: the two trainer shards covering this worker's rows.
        assert sources("model.layers.0.self_attn.q_proj.weight") == [str(2 * shard_rank), str(2 * shard_rank + 1)]
        # Split on dim 1: a column range of every trainer shard.
        assert sources("model.layers.0.mlp.down_proj.weight") == ["0", "1", "2", "3"]
        # Held whole: all of it, from every trainer shard.
        assert sources("model.layers.0.input_layernorm.weight") == ["0", "1", "2", "3"]
        # Every route moves exactly this worker's bytes (bf16: 2 bytes per element).
        for key, registration in registrations.items():
            moved = sum(transfer.nbytes for transfer in routes.lookup(key).transfers)
            assert moved == 2 * torch.Size(registration.tensor_slice.local_shape).numel()
