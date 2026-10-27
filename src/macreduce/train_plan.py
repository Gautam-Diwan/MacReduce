"""Plans for distributed QLoRA and explicit Neural Engine placement.

A plan is not a measurement. Probing an import does not show that the Neural
Engine ran. A missing explicit stack stays missing: this module does not
substitute the GPU.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass

from .finetune import (
    CURVE_ACCUMULATION,
    CURVE_BATCH_SIZE,
    CURVE_EPOCHS,
    CURVE_TRAIN_EXAMPLES,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    TRAIN_MODEL,
)

LOCAL = "local"
PEER = "peer"
EXPLICIT_ANE = ("aneforge", "orion", "maderix")
IMPLICIT_ANE = ("coreml",)
GPU_TRAINERS = ("mlx_gpu", "mlx_cpu")
TOPOLOGIES = (
    "single",
    "ring_allreduce",
    "heterogeneous_allreduce",
    "hierarchical_allreduce",
    "tree_allreduce",
    "parameter_server",
    "local_sgd",
    "alongside",
)
SCHEDULES = ("train_only", "ane_infer_while_gpu_trains")
QUESTIONS = ("baseline", "explicit_ane_trainer", "implicit_coreml", "overlap", "topology")

BACKENDS = {
    "mlx_gpu": {"role": "baseline", "module": "mlx"},
    "mlx_cpu": {"role": "reference", "module": "mlx"},
    "coreml": {"role": "implicit", "module": "coremltools"},
    "aneforge": {"role": "explicit", "module": "aneforge"},
    "orion": {"role": "explicit", "module": None},
    "maderix": {"role": "explicit", "module": None},
}


@dataclass(frozen=True)
class Seat:
    """One participant. ``host`` is ``local`` until a second Mac is named."""

    name: str
    host: str
    role: str
    train_backend: str | None = None
    infer_backend: str | None = None


@dataclass(frozen=True)
class TrainPlan:
    """One comparison. ``measured`` is always false on a plan that was only built."""

    name: str
    question: str
    topology: str
    schedule: str
    seats: tuple[Seat, ...]
    local_steps: int = 1

    def hosts(self) -> tuple[str, ...]:
        seen = []
        for seat in self.seats:
            if seat.host not in seen:
                seen.append(seat.host)
        return tuple(seen)

    def needs_second_mac(self) -> bool:
        return len(self.hosts()) > 1


def validate(plan: TrainPlan) -> None:
    """Reject a plan that smuggles in a backend or a topology it does not name."""

    if plan.question not in QUESTIONS:
        raise ValueError(f"unknown question {plan.question}")
    if plan.topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {plan.topology}")
    if plan.schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {plan.schedule}")
    if plan.local_steps < 1:
        raise ValueError("local steps must be positive")
    if plan.topology == "local_sgd" and plan.local_steps < 2:
        raise ValueError("local SGD needs more than one local step")
    if plan.topology != "local_sgd" and plan.local_steps != 1:
        raise ValueError("only local SGD varies the number of local steps")
    names = [seat.name for seat in plan.seats]
    if len(names) != len(set(names)):
        raise ValueError("seat names must be unique")
    trainers = [seat for seat in plan.seats if seat.role == "train"]
    servers = [seat for seat in plan.seats if seat.role == "server"]
    inferred = [seat for seat in plan.seats if seat.infer_backend]
    for seat in plan.seats:
        _check_seat(seat)
    if plan.topology == "single":
        if plan.needs_second_mac() or len(trainers) != 1 or servers:
            raise ValueError("single is one trainer on one Mac")
    if plan.topology == "ring_allreduce":
        if len(trainers) < 2:
            raise ValueError("ring all-reduce needs two trainers")
        if len({seat.train_backend for seat in trainers}) != 1:
            raise ValueError("ring all-reduce requires equal trainers")
    if plan.topology == "heterogeneous_allreduce":
        backends = {seat.train_backend for seat in trainers}
        gpu = backends & set(GPU_TRAINERS)
        ane = backends & set(EXPLICIT_ANE)
        if len(trainers) < 2 or not gpu or not ane:
            raise ValueError("heterogeneous all-reduce needs a GPU rank and an explicit Neural Engine rank")
    if plan.topology == "hierarchical_allreduce":
        if len(plan.hosts()) < 2:
            raise ValueError("hierarchical all-reduce needs two Macs")
        by_host: dict[str, set[str]] = {}
        for seat in trainers:
            by_host.setdefault(seat.host, set()).add(seat.train_backend or "")
        for host, backends in by_host.items():
            if not (backends & set(GPU_TRAINERS)) or not (backends & set(EXPLICIT_ANE)):
                raise ValueError(f"{host} must sum a GPU rank and an explicit Neural Engine rank first")
    if plan.topology == "tree_allreduce" and len(trainers) < 3:
        raise ValueError("a tree needs at least three trainers")
    if plan.topology == "local_sgd" and len(trainers) < 2:
        raise ValueError("local SGD needs two trainers")
    if plan.topology == "parameter_server" and (len(servers) != 1 or len(trainers) < 1):
        raise ValueError("a parameter server needs one server and a trainer")
    if plan.topology == "alongside" and (len(trainers) < 1 or not inferred):
        raise ValueError("alongside needs a GPU trainer and a Neural Engine inference seat")
    if plan.schedule == "train_only" and inferred:
        raise ValueError("train_only does not run inference")
    if plan.schedule == "ane_infer_while_gpu_trains":
        gpu = [seat for seat in trainers if seat.train_backend in GPU_TRAINERS]
        if not gpu or not inferred:
            raise ValueError("overlap needs a GPU trainer and a Neural Engine inference path")
    if plan.question == "explicit_ane_trainer":
        if not any(seat.train_backend in EXPLICIT_ANE for seat in trainers):
            raise ValueError("an explicit plan has to name ANEForge, Orion, or maderix")
    if plan.question == "baseline":
        if any(seat.train_backend not in GPU_TRAINERS for seat in trainers):
            raise ValueError("the baseline trains on MLX")


def comparison_plans() -> tuple[TrainPlan, ...]:
    """The comparisons the course proposal commits to. None is a result."""

    plans = (
        _plan("gpu_one_mac", "baseline", "single", "train_only", (_trainer("m4", LOCAL, "mlx_gpu"),)),
        _plan(
            "aneforge_one_mac",
            "explicit_ane_trainer",
            "single",
            "train_only",
            (_trainer("m4", LOCAL, "aneforge"),),
        ),
        _plan(
            "orion_one_mac",
            "explicit_ane_trainer",
            "single",
            "train_only",
            (_trainer("m4", LOCAL, "orion"),),
        ),
        _plan(
            "maderix_one_mac",
            "explicit_ane_trainer",
            "single",
            "train_only",
            (_trainer("m4", LOCAL, "maderix"),),
        ),
        _plan(
            "coreml_one_mac",
            "implicit_coreml",
            "single",
            "train_only",
            (_trainer("m4", LOCAL, "coreml"),),
        ),
        _plan(
            "overlap_one_mac",
            "overlap",
            "alongside",
            "ane_infer_while_gpu_trains",
            (_trainer("m4", LOCAL, "mlx_gpu", infer_backend="aneforge"),),
        ),
        _plan(
            "heterogeneous_one_mac",
            "topology",
            "heterogeneous_allreduce",
            "train_only",
            (_trainer("gpu", LOCAL, "mlx_gpu"), _trainer("ane", LOCAL, "aneforge")),
        ),
        _plan(
            "ring_two_mac",
            "topology",
            "ring_allreduce",
            "train_only",
            (_trainer("a", LOCAL, "mlx_gpu"), _trainer("b", PEER, "mlx_gpu")),
        ),
        _plan(
            "heterogeneous_two_mac",
            "topology",
            "heterogeneous_allreduce",
            "train_only",
            (_trainer("gpu", LOCAL, "mlx_gpu"), _trainer("ane", PEER, "aneforge")),
        ),
        _plan(
            "hierarchical_two_mac",
            "topology",
            "hierarchical_allreduce",
            "train_only",
            (
                _trainer("gpu0", LOCAL, "mlx_gpu"),
                _trainer("ane0", LOCAL, "aneforge"),
                _trainer("gpu1", PEER, "mlx_gpu"),
                _trainer("ane1", PEER, "aneforge"),
            ),
        ),
        _plan(
            "tree_four_ranks",
            "topology",
            "tree_allreduce",
            "train_only",
            (
                _trainer("gpu0", LOCAL, "mlx_gpu"),
                _trainer("ane0", LOCAL, "aneforge"),
                _trainer("gpu1", PEER, "mlx_gpu"),
                _trainer("ane1", PEER, "aneforge"),
            ),
        ),
        _plan(
            "alongside_two_mac",
            "overlap",
            "alongside",
            "ane_infer_while_gpu_trains",
            (
                _trainer("gpu", LOCAL, "mlx_gpu"),
                Seat("ane", PEER, "infer", infer_backend="aneforge"),
            ),
        ),
    )
    for plan in plans:
        validate(plan)
    return plans


def probe_backends() -> list[dict]:
    """Say which stacks import. Importing one is not Neural Engine residency."""

    rows = []
    for name, spec in BACKENDS.items():
        module = spec["module"]
        row = {
            "backend": name,
            "role": spec["role"],
            "module": module,
            "explicit": name in EXPLICIT_ANE,
            "placement_verified": False,
            "measured": False,
        }
        if module is None:
            row["status"] = "unpinned"
            rows.append(row)
            continue
        try:
            importlib.import_module(module)
        except ImportError as exc:
            row["status"] = "missing"
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["status"] = "importable"
        rows.append(row)
    return rows


def report() -> dict:
    """The plan and the probe. ``status`` is planned, not a training result."""

    plans = comparison_plans()
    return {
        "status": "planned",
        "measured": False,
        "dataset": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
        },
        "model": TRAIN_MODEL,
        "baseline": "gpu_one_mac",
        "distributed_baseline": "ring_two_mac",
        "curve": {
            "train_examples": CURVE_TRAIN_EXAMPLES,
            "epochs": CURVE_EPOCHS,
            "batch_size": CURVE_BATCH_SIZE,
            "grad_accumulation_steps": CURVE_ACCUMULATION,
        },
        "backends": probe_backends(),
        "plans": [_public(plan) for plan in plans],
    }


def _trainer(name: str, host: str, backend: str, infer_backend: str | None = None) -> Seat:
    return Seat(name, host, "train", train_backend=backend, infer_backend=infer_backend)


def _plan(name, question, topology, schedule, seats, local_steps=1) -> TrainPlan:
    return TrainPlan(name, question, topology, schedule, seats, local_steps)


def _check_seat(seat: Seat) -> None:
    if seat.role not in ("train", "server", "infer"):
        raise ValueError(f"unknown role {seat.role}")
    allowed = (*GPU_TRAINERS, *EXPLICIT_ANE, *IMPLICIT_ANE)
    if seat.role == "train" and seat.train_backend not in allowed:
        raise ValueError(f"unknown train backend {seat.train_backend}")
    if seat.role != "train" and seat.train_backend is not None:
        raise ValueError("only a trainer has a train backend")
    if seat.infer_backend is not None and seat.infer_backend not in (*EXPLICIT_ANE, *IMPLICIT_ANE):
        raise ValueError(f"unknown infer backend {seat.infer_backend}")
    if seat.role == "infer" and seat.infer_backend is None:
        raise ValueError("an inference seat needs a Neural Engine backend")


def _public(plan: TrainPlan) -> dict:
    body = asdict(plan)
    body["hosts"] = list(plan.hosts())
    body["needs_second_mac"] = plan.needs_second_mac()
    body["measured"] = False
    return body
