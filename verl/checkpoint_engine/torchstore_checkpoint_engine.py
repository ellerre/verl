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

"""TorchStore checkpoint backend: trainer DTensor shards in, destination-local shards out.

Every trainer rank publishes its parameters to TorchStore as it holds them: one
local DTensor shard per parameter, stored with its position in the full tensor.
Every rollout worker asks for the slice of each parameter that its inference rank
holds, and TorchStore fills it from the trainer shards that overlap it. No rank
materializes a full tensor, and the layouts of the two sides need not match.

``mode="store"`` moves the weights through TorchStore storage volumes (host memory
next to the trainers); ``mode="direct"`` lets rollout workers read trainer GPU
memory through Monarch RDMA buffers, without a volume.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import pickle
import socket
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, AsyncGenerator, Coroutine, Generator, Sequence

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.models.transformers.hf_dense_decoder_tp import infer_dense_decoder_tp_shard_dim
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_KV_PREFIX = "verl/torchstore"
# Destination tensors are views into one buffer; any dtype can view an offset
# that is a multiple of this.
_ALIGNMENT = 64
_MODES = ("store", "direct")
_STRATEGIES = ("local_rank", "host")
# Monarch's worker port must stay clear of the bootstrap store's.
_MIN_PORT_DISTANCE = 100


def _require_torchstore() -> None:
    try:
        import torchstore  # noqa: F401
        from monarch.job.spmd import StoreJob  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "checkpoint_engine.backend='torchstore' requires torchstore and a torchmonarch "
            "release that provides monarch.job.spmd.StoreJob (0.7.0 or later)"
        ) from exc


def _require_direct_rdma() -> None:
    from torchstore.transport.monarch_rdma import monarch_rdma_transport_available

    if not monarch_rdma_transport_available():
        raise RuntimeError(
            "torchstore direct mode is unavailable on this host: it reads trainer GPU memory through "
            "Monarch RDMA buffers, which need an ibverbs device; use mode='store'"
        )


@dataclass(frozen=True)
class TorchStoreMasterMetadata:
    """Bootstrap record created by the master trainer rank.

    Attributes:
        host: Address of the bootstrap ``TCPStore`` that the master rank serves.
        store_port: Port of the bootstrap ``TCPStore``.
        monarch_port: Port every host's Monarch worker listens on.
        source_dp: Number of replicated groups in the trainer mesh.
        source_shard_size: Number of trainer ranks per replicated group.
        destination_dp: Number of rollout replicas.
        destination_shard_size: Number of rollout workers per replica.
        mode: ``"store"`` or ``"direct"``.
        strategy: ``"local_rank"`` (one volume per rank) or ``"host"`` (one per host).
        routing: Whether clients plan their reads once from registered layouts.
    """

    host: str
    store_port: int
    monarch_port: int
    source_dp: int
    source_shard_size: int
    destination_dp: int
    destination_shard_size: int
    mode: str
    strategy: str
    routing: bool


@dataclass(frozen=True)
class TorchStoreWorkerInfo:
    """What ``prepare`` returns on every worker.

    Attributes:
        hostname: Host of the worker; ``build_topology`` derives the bootstrap ranks from it.
        master: Bootstrap record on the master trainer rank, ``None`` elsewhere.
    """

    hostname: str
    master: TorchStoreMasterMetadata | None = None


def assign_spmd_ranks(
    trainer_hosts: Sequence[str], rollout_hosts: Sequence[str]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Number trainer and rollout workers as the torchrun-style ranks TorchStore boots from.

    TorchStore starts one Monarch worker per host, from the process whose local rank
    is 0, and takes the host of a rank to be ``rank // local_world_size``. So every
    host must hold the same number of workers, and trainer rank ``i`` must keep rank
    ``i``: its volume (one per rank) and its routing identity are named after it.

    Args:
        trainer_hosts: Hostname of each trainer worker, in trainer rank order.
        rollout_hosts: Hostname of each rollout worker, in rollout worker order.

    Returns:
        ``(rank, local_rank)`` per trainer worker, the same per rollout worker, and
        the number of workers per host.

    Raises:
        ValueError: If hosts hold different numbers of workers, or trainer ranks do
            not fill their hosts in rank order.
    """

    hosts = [*trainer_hosts, *rollout_hosts]
    order = list(dict.fromkeys(hosts))
    counts = Counter(hosts)
    if len(set(counts.values())) != 1:
        raise ValueError(f"TorchStore needs the same number of workers on every host, got {dict(counts)}")
    local_world_size = counts[order[0]]
    next_local_rank = dict.fromkeys(order, 0)
    ranks = []
    for host in hosts:
        local_rank = next_local_rank[host]
        next_local_rank[host] += 1
        ranks.append((order.index(host) * local_world_size + local_rank, local_rank))
    trainer_ranks = ranks[: len(trainer_hosts)]
    for index, (rank, _) in enumerate(trainer_ranks):
        if rank != index:
            raise ValueError(
                "TorchStore needs trainer ranks to fill their hosts in rank order; "
                f"trainer rank {index} on {trainer_hosts[index]!r} would get bootstrap rank {rank}"
            )
    return trainer_ranks, ranks[len(trainer_hosts) :], local_world_size


@dataclass(frozen=True)
class DestinationWeight:
    """One weight as a rollout worker holds it.

    Attributes:
        name: Parameter name.
        global_shape: Shape of the full tensor.
        dtype: Dtype the weight is stored and transferred in.
        shard_dim: Dimension the rollout replica shards the weight on, ``None`` if
            every worker of the replica holds it whole.
        local_shape: Shape of this worker's part.
        offset: Byte offset of this worker's part in the destination buffer.
        nbytes: Byte size of this worker's part.
    """

    name: str
    global_shape: tuple[int, ...]
    dtype: torch.dtype
    shard_dim: int | None
    local_shape: tuple[int, ...]
    offset: int
    nbytes: int


def destination_layout(
    manifest: Sequence[tuple[str, Sequence[int], str]], shard_size: int
) -> tuple[list[DestinationWeight], int]:
    """Lay the weights of one rollout worker out in a single buffer.

    Args:
        manifest: ``(name, global_shape, dtype_name)`` per weight, in publication order.
        shard_size: Number of workers a rollout replica shards its weights over.

    Returns:
        The weights with their local shapes and buffer offsets, and the buffer size in bytes.

    Raises:
        ValueError: If a sharded dimension does not divide by ``shard_size``.
        NotImplementedError: If the placement of a weight on the replica is unknown.
    """

    weights = []
    offset = 0
    for name, shape, dtype_name in manifest:
        global_shape = tuple(int(size) for size in shape)
        dtype = getattr(torch, dtype_name)
        shard_dim = infer_dense_decoder_tp_shard_dim(name) if shard_size > 1 else None
        local_shape = global_shape
        if shard_dim is not None:
            if global_shape[shard_dim] % shard_size:
                raise ValueError(
                    f"{name}: dimension {shard_dim} of {global_shape} does not divide by {shard_size} rollout workers"
                )
            local_shape = (
                *global_shape[:shard_dim],
                global_shape[shard_dim] // shard_size,
                *global_shape[shard_dim + 1 :],
            )
        nbytes = math.prod(local_shape) * dtype.itemsize
        weights.append(DestinationWeight(name, global_shape, dtype, shard_dim, local_shape, offset, nbytes))
        offset = -(-(offset + nbytes) // _ALIGNMENT) * _ALIGNMENT
    return weights, offset


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _virtual_mesh(size: int, rank: int, device_type: str) -> Any:
    """A one-dimensional device mesh that knows this rank's position and has no process group.

    TorchStore only reads a DTensor's mesh shape, coordinate and placements, so the
    rollout workers of a replica need no collective to describe their shards.
    """

    from torch.distributed.device_mesh import DeviceMesh

    try:
        return DeviceMesh(device_type, torch.arange(size), _init_backend=False, _rank=rank)
    except TypeError as exc:
        raise NotImplementedError(
            "the torchstore backend needs a torch whose DeviceMesh takes its rank without a process group"
        ) from exc


def destination_tensors(
    weights: Sequence[DestinationWeight], buffer: torch.Tensor, shard_size: int, shard_rank: int
) -> tuple[dict[str, torch.Tensor], list[tuple[str, torch.Tensor]]]:
    """Describe one rollout worker's weights to TorchStore, as views into its buffer.

    Args:
        weights: The worker's weights, from ``destination_layout``.
        buffer: Byte buffer of the size ``destination_layout`` returned.
        shard_size: Number of workers the rollout replica shards its weights over.
        shard_rank: This worker's position among them.

    Returns:
        What TorchStore fills, by name (a DTensor over a virtual mesh for a sharded
        weight, so that TorchStore reads this worker's position off it; a plain
        tensor for a weight held whole), and the ``(name, local tensor)`` pairs the
        engine yields. Both are views into ``buffer``.
    """

    from torch.distributed.tensor import DTensor, Shard

    mesh = _virtual_mesh(shard_size, shard_rank, buffer.device.type) if shard_size > 1 else None
    destination: dict[str, torch.Tensor] = {}
    outputs = []
    for weight in weights:
        local = buffer[weight.offset : weight.offset + weight.nbytes].view(weight.dtype).view(weight.local_shape)
        target = local
        if weight.shard_dim is not None:
            target = DTensor.from_local(local, mesh, (Shard(weight.shard_dim),), run_check=False)
            if tuple(target.shape) != weight.global_shape or target.to_local().data_ptr() != local.data_ptr():
                raise RuntimeError(f"{weight.name}: destination shard does not describe {weight.global_shape}")
        destination[weight.name] = target
        outputs.append((weight.name, local))
    return destination, outputs


class _EventLoopThread:
    """One event loop on its own thread: every TorchStore call of a process runs on it."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._main, name="verl-torchstore", daemon=True)
        self._thread.start()

    def _main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine: Coroutine, timeout: float | None = None) -> Any:
        """Run a coroutine on the loop and block until it ends."""

        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)

    async def arun(self, coroutine: Coroutine) -> Any:
        """Run a coroutine on the loop without blocking the caller's event loop."""

        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, self._loop))

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


@CheckpointEngineRegistry.register("torchstore")
class TorchStoreCheckpointEngine(CheckpointEngine):
    """Transfer trainer DTensor shards into destination rank-local shards through TorchStore.

    Trainer and rollout ranks form two meshes with shapes
    ``(source_dp, source_shard_size)`` and ``(destination_dp, destination_shard_size)``,
    as for the ``nccl_m2n`` backend. TorchStore reshards between them: a rollout
    worker reads only the slice it holds, from the trainer shards overlapping it.

    The trainer side consumes ``(name, DTensor)`` pairs
    (``get_per_tensor_param_dtensor``); the rollout side yields ``(name, tensor)``
    pairs already partitioned for its worker, as views into one buffer that the next
    update overwrites.

    TorchStore is started once, in the first ``init_process_group``, from every
    worker at the same time (its SPMD bootstrap), and lives until the processes end.
    Layouts are registered and destinations allocated in the first update. Because
    the manager starts ``send_weights`` and ``receive_weights`` together, each
    trainer rank marks its put as complete in the bootstrap store and rollout
    workers wait for every mark before they read.

    Args:
        bucket_size: Common checkpoint-engine bucket size in bytes. Unused: kept for
            checkpoint-engine interface compatibility.
        is_master: Whether this trainer rank creates the bootstrap record. Exactly
            one trainer rank must be the master.
        source_dp: Number of replicated groups in the trainer mesh.
        source_shard_size: Number of trainer ranks per replicated group.
        destination_dp: Number of rollout replicas.
        destination_shard_size: Number of rollout workers per replica. A consumer
            such as vLLM interprets it as its tensor-parallel size.
        mode: ``"store"`` to go through storage volumes, ``"direct"`` to read trainer
            GPU memory through Monarch RDMA buffers.
        strategy: Volume placement in store mode: ``"local_rank"`` for one volume per
            rank, ``"host"`` for one per host.
        routing: Plan every read once from layouts registered with a coordinator,
            instead of asking the controller where each key lives on every update.
            TorchStore routes to one volume per publisher rank, so it requires
            ``strategy="local_rank"``. Defaults to on for that strategy in store mode.
        key: TorchStore namespace of the published state dict.
        store_name: Name of the TorchStore instance.
        transfer_dtype: Dtype floating-point weights are stored and transferred in,
            ``None`` to keep the trainer's.
        control_transport: Monarch's control-plane transport. ``"ipc"`` only works
            on a single host.
        require_transports: If set, the names of the TorchStore data transports a
            worker may resolve for any volume (e.g. ``["SharedMemory", "NIXL"]``);
            anything else fails ``init_process_group``.
        pin_destination: Keep the destination buffer registered with NIXL between
            updates instead of registering it for every read.
        store_env: Environment variables to set in this process before TorchStore
            starts (``None`` or an empty value unsets one). The storage volumes
            this process spawns inherit them, and so does any transport that
            starts later in this process; libraries already initialized here,
            such as the trainer's NCCL, are not affected. For transports whose
            fabric settings differ from the ones NCCL needs.
        timeout_s: Limit, in seconds, on the bootstrap and on each wait for the
            trainers' puts.
    """

    wire_format = "rank_local_named_tensors"

    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        source_dp: int | None = None,
        source_shard_size: int | None = None,
        destination_dp: int | None = None,
        destination_shard_size: int | None = None,
        mode: str = "store",
        strategy: str = "local_rank",
        routing: bool | None = None,
        key: str = "policy",
        store_name: str = "verl_torchstore",
        transfer_dtype: str | None = "bfloat16",
        control_transport: str = "tcp",
        require_transports: Sequence[str] | None = None,
        pin_destination: bool = False,
        store_env: dict[str, str | None] | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        topology = {
            "source_dp": source_dp,
            "source_shard_size": source_shard_size,
            "destination_dp": destination_dp,
            "destination_shard_size": destination_shard_size,
        }
        missing = [name for name, value in topology.items() if value is None]
        if missing:
            raise ValueError(f"TorchStore topology requires explicit values for {', '.join(missing)}")
        if mode not in _MODES:
            raise ValueError(f"TorchStore mode must be one of {_MODES}, got {mode!r}")
        if strategy not in _STRATEGIES:
            raise ValueError(f"TorchStore strategy must be one of {_STRATEGIES}, got {strategy!r}")
        if routing is None:
            routing = mode == "store" and strategy == "local_rank"
        if routing and (mode != "store" or strategy != "local_rank"):
            raise ValueError("TorchStore routing requires mode='store' and strategy='local_rank'")

        self.bucket_size = int(bucket_size)
        self.is_master = bool(is_master)
        self.source_dp = int(source_dp)
        self.source_shard_size = int(source_shard_size)
        self.destination_dp = int(destination_dp)
        self.destination_shard_size = int(destination_shard_size)
        if min(self.source_dp, self.source_shard_size, self.destination_dp, self.destination_shard_size) <= 0:
            raise ValueError("TorchStore mesh sizes must be positive")
        self.mode = mode
        self.strategy = strategy
        self.routing = bool(routing)
        self.key = key
        self.store_name = store_name
        self.transfer_dtype = None if transfer_dtype is None else getattr(torch, transfer_dtype)
        self.control_transport = control_transport
        self.require_transports = None if require_transports is None else [str(name) for name in require_transports]
        self.pin_destination = bool(pin_destination)
        self.store_env = {
            str(name): None if value is None or str(value) == "" else str(value)
            for name, value in dict(store_env or {}).items()
        }
        self.timeout_s = float(timeout_s)

        self.source_world_size = self.source_dp * self.source_shard_size
        self.destination_world_size = self.destination_dp * self.destination_shard_size
        self.rank: int | None = None
        self.role: str | None = None
        self.index: int | None = None
        self._info: TorchStoreWorkerInfo | None = None
        self._master: TorchStoreMasterMetadata | None = None
        self._topology: tuple | None = None
        self._loop: _EventLoopThread | None = None
        self._client: Any = None
        self._kv: Any = None
        # Data transports this worker resolved, e.g. {"local=SharedMemory": 4, "remote=NIXL": 4}.
        self.transports: dict[str, int] | None = None
        self._sync_id = 0
        # Trainer: storage key and transfer dtype per weight, CUDA devices to fence.
        self._storage_keys: dict[str, str] | None = None
        self._transfer_dtypes: dict[str, torch.dtype] = {}
        self._devices: list[torch.device] = []
        # Rollout worker: destination buffer, what TorchStore fills, what is yielded.
        self._buffer: torch.Tensor | None = None
        self._destination: dict[str, torch.Tensor] | None = None
        self._entries: dict[str, torch.Tensor] | None = None
        self._outputs: list[tuple[str, torch.Tensor]] | None = None
        self._pin: tuple[Any, Any] | None = None

    def prepare(self) -> TorchStoreWorkerInfo:
        """Report this worker's host and, on the master trainer rank, the bootstrap record.

        Returns:
            The same cached record on every call.

        Raises:
            ImportError: On the master, if TorchStore or its Monarch bootstrap is missing.
            RuntimeError: On the master, if direct mode is unavailable on this host.
        """

        if self._info is None:
            master = None
            if self.is_master:
                _require_torchstore()
                if self.mode == "direct":
                    _require_direct_rdma()
                host = ray.util.get_node_ip_address().strip("[]")
                store_port, _ = get_free_port(host)
                monarch_port = store_port
                while abs(monarch_port - store_port) < _MIN_PORT_DISTANCE:
                    monarch_port, _ = get_free_port(host)
                master = TorchStoreMasterMetadata(
                    host,
                    store_port,
                    monarch_port,
                    self.source_dp,
                    self.source_shard_size,
                    self.destination_dp,
                    self.destination_shard_size,
                    self.mode,
                    self.strategy,
                    self.routing,
                )
            self._info = TorchStoreWorkerInfo(socket.gethostname(), master)
        return self._info

    @classmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, metadata: list[Any]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Assign every worker its TorchStore bootstrap rank.

        Args:
            trainer_world_size: Number of trainer workers.
            rollout_world_size: Number of rollout workers.
            metadata: Results of ``prepare`` ordered with trainer workers first.

        Returns:
            Trainer and rollout keyword-argument maps for ``init_process_group``.
            Every value is a per-worker list suitable for worker-group dispatch.

        Raises:
            ValueError: If there is not exactly one trainer master, a worker count
                does not match the configured meshes, or the workers' hosts do not
                fit TorchStore's bootstrap (see ``assign_spmd_ranks``).
        """

        infos = list(metadata)
        if len(infos) != trainer_world_size + rollout_world_size or not all(
            isinstance(info, TorchStoreWorkerInfo) for info in infos
        ):
            raise ValueError("TorchStore topology needs the prepare() record of every worker")
        masters = [info.master for info in infos[:trainer_world_size] if info.master is not None]
        if len(masters) != 1:
            raise ValueError(f"TorchStore requires exactly one trainer master, got {len(masters)}")
        master = masters[0]
        source_size = master.source_dp * master.source_shard_size
        destination_size = master.destination_dp * master.destination_shard_size
        if trainer_world_size != source_size or rollout_world_size != destination_size:
            raise ValueError(
                f"TorchStore requires exactly {source_size} trainer and {destination_size} rollout ranks, "
                f"got {trainer_world_size} and {rollout_world_size}"
            )
        trainer_ranks, rollout_ranks, local_world_size = assign_spmd_ranks(
            [info.hostname for info in infos[:trainer_world_size]],
            [info.hostname for info in infos[trainer_world_size:]],
        )
        world_size = source_size + destination_size

        def kwargs(ranks: list[tuple[int, int]], role: str) -> dict[str, list[Any]]:
            count = len(ranks)
            return {
                "rank": [rank for rank, _ in ranks],
                "local_rank": [local_rank for _, local_rank in ranks],
                "world_size": [world_size] * count,
                "local_world_size": [local_world_size] * count,
                "master_metadata": [master] * count,
                "role": [role] * count,
                "index": list(range(count)),
            }

        return kwargs(trainer_ranks, "source"), kwargs(rollout_ranks, "destination")

    def init_process_group(
        self,
        rank: int,
        local_rank: int,
        world_size: int,
        local_world_size: int,
        master_metadata: TorchStoreMasterMetadata,
        role: str,
        index: int,
    ) -> None:
        """Start TorchStore in this process, once; later calls only check the topology.

        Must run on every trainer and rollout worker at the same time: TorchStore's
        bootstrap is collective.

        Args:
            rank: This worker's TorchStore bootstrap rank.
            local_rank: This worker's position among the workers of its host.
            world_size: Total number of trainer and rollout workers.
            local_world_size: Number of workers per host.
            master_metadata: Bootstrap record returned by the master trainer rank.
            role: ``"source"`` for a trainer rank, ``"destination"`` for a rollout worker.
            index: Trainer rank, or position of the rollout worker among all rollout
                workers (replica by replica).

        Raises:
            ValueError: If the role, world size or topology does not match this engine.
            RuntimeError: If TorchStore is already running with a different topology,
                a volume is not where its trainer rank is, or a transport outside
                ``require_transports`` is resolved.
        """

        topology = (int(rank), int(local_rank), int(world_size), int(local_world_size), role, int(index))
        if self._topology is not None:
            if topology != self._topology or master_metadata != self._master:
                raise RuntimeError("cannot reuse a TorchStore session with a different topology")
            return

        if role not in {"source", "destination"}:
            raise ValueError(f"TorchStore role must be source or destination, got {role!r}")
        if world_size != self.source_world_size + self.destination_world_size:
            raise ValueError(
                f"runtime world_size={world_size}, expected {self.source_world_size + self.destination_world_size}"
            )
        expected = (
            self.source_dp,
            self.source_shard_size,
            self.destination_dp,
            self.destination_shard_size,
            self.mode,
            self.strategy,
            self.routing,
        )
        actual = (
            master_metadata.source_dp,
            master_metadata.source_shard_size,
            master_metadata.destination_dp,
            master_metadata.destination_shard_size,
            master_metadata.mode,
            master_metadata.strategy,
            master_metadata.routing,
        )
        if actual != expected:
            raise ValueError(f"TorchStore configuration {actual} does not match local config {expected}")
        if role == "source" and rank != index:
            raise ValueError(f"trainer rank {index} must keep TorchStore rank {index}, got {rank}")

        _require_torchstore()
        self.rank, self.role, self.index = int(rank), role, int(index)
        self._master = master_metadata
        for name, value in self.store_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._loop = _EventLoopThread()
        self._loop.run(self._bootstrap(int(local_rank), int(world_size), int(local_world_size)))
        self._topology = topology

    async def _bootstrap(self, local_rank: int, world_size: int, local_world_size: int) -> None:
        import torchstore as ts
        from torch.distributed import TCPStore
        from torchstore.spmd import SPMDEnv

        master = self._master
        timeout = timedelta(seconds=self.timeout_s)
        strategy = ts.LocalRankStrategy() if master.strategy == "local_rank" else ts.HostStrategy()
        await ts.initialize_spmd(
            strategy=strategy,
            store_name=self.store_name,
            env=SPMDEnv(
                rank=self.rank,
                local_rank=local_rank,
                world_size=world_size,
                local_world_size=local_world_size,
                master_addr=master.host,
                master_port=master.store_port,
            ),
            rendezvous_timeout=timeout,
            transport=self.control_transport,
            monarch_port=master.monarch_port,
        )
        # A second connection to the bootstrap store TorchStore's rank 0 serves:
        # it carries the manifest, the routing coordinator and the commit marks.
        self._kv = TCPStore(master.host, master.store_port, is_master=False, wait_for_workers=False, timeout=timeout)
        client = await ts.client(self.store_name)
        self._check_volumes(client.strategy)
        if master.routing:
            client = await self._routing_client(client.strategy)
        self._client = client

    def _check_volumes(self, strategy: Any) -> None:
        """Check where this trainer rank's volume is and which transports this worker resolves."""

        from torchstore.transport import TransportType, get_available_transport
        from torchstore.utils import get_local_hostname

        hostname = get_local_hostname()
        volume_hosts = strategy.volume_id_to_hostname
        if self.role == "source" and self._master.mode == "store":
            volume_id = str(self.rank) if self._master.strategy == "local_rank" else hostname
            if volume_hosts.get(volume_id) != hostname:
                raise RuntimeError(
                    f"TorchStore volume {volume_id!r} is on {volume_hosts.get(volume_id)!r}, "
                    f"not on the host of trainer rank {self.index} ({hostname!r})"
                )
            if not self._master.routing and strategy.select_storage_volume().volume_id != volume_id:
                raise RuntimeError(
                    f"trainer rank {self.index} would put into volume "
                    f"{strategy.select_storage_volume().volume_id!r}, expected {volume_id!r}"
                )
        resolved = {}
        for volume_id in volume_hosts:
            volume = strategy.get_storage_volume(volume_id)
            transport = volume.default_transport_type
            if transport == TransportType.Unset:
                transport = get_available_transport(volume)
            resolved[volume_id] = transport.name
        summary = Counter(
            f"{'local' if volume_hosts[volume_id] == hostname else 'remote'}={name}"
            for volume_id, name in resolved.items()
        )
        self.transports = dict(summary)
        logger.info(
            "torchstore rank %d (%s %d) on %s: %d volumes, transports %s",
            self.rank,
            self.role,
            self.index,
            hostname,
            len(resolved),
            dict(summary),
        )
        if self.require_transports is not None and not set(resolved.values()) <= set(self.require_transports):
            raise RuntimeError(
                f"TorchStore resolved transports {dict(summary)} on {hostname}, allowed: {self.require_transports}"
            )

    async def _routing_client(self, strategy: Any) -> Any:
        """Build this worker's routing client around one coordinator shared through the bootstrap store.

        A publisher's routing identity is the id of the volume it puts into, which
        for one volume per rank is its TorchStore rank.
        """

        import cloudpickle
        from monarch.actor import get_or_spawn_controller
        from torchstore import RankRole
        from torchstore.routing.client import RoutingClient
        from torchstore.routing.coordinator import MonarchControlPlane, RoutingCoordinator

        key = f"{_KV_PREFIX}/coordinator"
        if self.rank == 0:
            coordinator = await get_or_spawn_controller(f"{self.store_name}_routing", RoutingCoordinator)
            plane = MonarchControlPlane(coordinator)
            await plane.init(
                publishers={str(rank) for rank in range(self.source_world_size)},
                strategy=strategy,
            )
            self._kv.set(key, cloudpickle.dumps(coordinator))
        else:
            plane = MonarchControlPlane(cloudpickle.loads(self._kv.get(key)))
        if self.role == "source":
            return RoutingClient(str(self.rank), RankRole.PUBLISHER, plane, await plane.strategy())
        return RoutingClient(f"requester/{self.rank}", RankRole.REQUESTER, plane, await plane.strategy())

    def _require_role(self, role: str) -> None:
        if self._topology is None or self._client is None:
            raise RuntimeError("TorchStore process group is not initialized")
        if self.role != role:
            raise RuntimeError(f"invalid TorchStore state for role={self.role!r}, expected {role!r}")

    async def send_weights(self, weights: Generator, global_steps: int | None = None) -> dict:
        """Publish this trainer rank's local shards and mark the put as complete.

        Args:
            weights: Generator yielding ``(name, DTensor)`` pairs, or plain tensors
                for weights this rank holds whole. The same names on every update.
            global_steps: Optional trainer step associated with the update. Reserved
                for checkpoint-engine interface compatibility.

        Returns:
            An empty metrics dictionary once every shard is stored.
        """

        del global_steps
        self._require_role("source")
        await self._loop.arun(self._send(dict(weights)))
        return {}

    @torch.no_grad()
    async def _send(self, state: dict[str, torch.Tensor]) -> None:
        sync_id = self._sync_id
        self._sync_id += 1
        direct = self._master.mode == "direct"
        if self._storage_keys is None:
            wire = {
                name: self.transfer_dtype
                if self.transfer_dtype is not None and tensor.is_floating_point()
                else tensor.dtype
                for name, tensor in state.items()
            }
            if self.is_master:
                manifest = [(name, tuple(tensor.shape), _dtype_name(wire[name])) for name, tensor in state.items()]
                self._kv.set(f"{_KV_PREFIX}/manifest", pickle.dumps(manifest))
            if not direct:
                await self._client.register_layout(state, self.key, transfer_dtype=self.transfer_dtype)
            storage_keys = {name: f"{self.key}/{name}" for name in state}
            self._transfer_dtypes = {
                storage_keys[name]: wire[name] for name, tensor in state.items() if wire[name] != tensor.dtype
            }
            self._devices = list({tensor.device for tensor in state.values() if tensor.is_cuda})
            self._storage_keys = storage_keys
        elif state.keys() != self._storage_keys.keys():
            raise RuntimeError("the published weight names changed between updates")

        await self._put(state)
        self._kv.set(f"{_KV_PREFIX}/commit/{sync_id}/{self.index}", b"1")

    async def _put(self, state: dict[str, torch.Tensor]) -> None:
        """Store this rank's shards, or in direct mode refresh what readers pull from."""

        import torchstore as ts

        # Transports copy from, or let peers read, parameter memory outside CUDA stream order.
        self._fence()
        if self._master.mode == "direct":
            await ts.put_state_dict(
                state, self.key, store_name=self.store_name, direct_rdma=True, transfer_dtype=self.transfer_dtype
            )
            # The put refreshes the cast staging buffers that readers pull from.
            self._fence()
        else:
            entries = {self._storage_keys[name]: tensor for name, tensor in state.items()}
            await self._client.put_batch(entries, transfer_dtypes=self._transfer_dtypes or None)

    def _fence(self) -> None:
        for device in self._devices:
            torch.cuda.current_stream(device).synchronize()

    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Read this rollout worker's part of every weight once all trainers have put.

        Args:
            global_steps: Optional trainer step associated with the update. Reserved
                for checkpoint-engine interface compatibility.

        Yields:
            The parameter name and its destination rank-local tensor, in publication
            order. The tensors are views into one buffer that the next update
            overwrites, so the consumer must copy them before then.
        """

        del global_steps
        self._require_role("destination")
        # The caller's device: the engine's own thread does not inherit it.
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        for name, tensor in await self._loop.arun(self._receive(device)):
            yield name, tensor

    @torch.no_grad()
    async def _receive(self, device: torch.device) -> list[tuple[str, torch.Tensor]]:
        sync_id = self._sync_id
        self._sync_id += 1
        if self._outputs is None:
            await self._prepare_destination(device)
        await self._wait_for_puts(sync_id)
        await self._read()
        return self._outputs

    async def _wait_for_puts(self, sync_id: int) -> None:
        """Block until every trainer rank has marked this update's put as complete."""

        marks = [f"{_KV_PREFIX}/commit/{sync_id}/{index}" for index in range(self.source_world_size)]
        await asyncio.get_running_loop().run_in_executor(
            None, self._kv.wait, marks, timedelta(seconds=self.timeout_s)
        )

    async def _read(self) -> None:
        """Fill the destination buffer with this worker's part of every weight."""

        import torchstore as ts

        if self._master.mode == "direct":
            await ts.get_state_dict(
                self.key, user_state_dict=self._destination, store_name=self.store_name, direct_rdma=True
            )
        else:
            await self._client.get_batch(self._entries)

    async def _prepare_destination(self, device: torch.device) -> None:
        """Allocate this worker's buffer and describe its shards to TorchStore, once."""

        shard_size = self.destination_shard_size
        manifest = pickle.loads(self._kv.get(f"{_KV_PREFIX}/manifest"))
        weights, nbytes = destination_layout(manifest, shard_size)
        buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
        destination, outputs = destination_tensors(weights, buffer, shard_size, self.index % shard_size)
        if self._master.mode == "store":
            await self._client.get_layouts(destination, self.key, strict=True)
        self._entries = {f"{self.key}/{name}": target for name, target in destination.items()}
        self._buffer, self._destination, self._outputs = buffer, destination, outputs
        if self.pin_destination and self._master.mode == "store":
            self._pin_buffer()

    def _pin_buffer(self) -> None:
        """Hold one NIXL registration of the destination buffer for the life of the process."""

        from torchstore.transport.nixl import NixlAgentCache, nixl_available

        if nixl_available() and self._buffer.is_cuda:
            cache = self._client.strategy.transport_context.get(NixlAgentCache)
            self._pin = (cache, cache.acquire(self._buffer))

    def finalize(self) -> None:
        """Nothing to release: a put and a read are complete when their calls return."""

    def close(self) -> None:
        """Stop this engine's event loop. TorchStore's processes end with their parents."""

        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        self._client = None
        self._topology = None
