"""领域模型：站点、网络链路、数据集、租户、作业组与预留。

时间以整数槽计量（见 :mod:`compute_network_scheduler.clock`）。所有模型都可
序列化为纯数据结构，供 JSON 仓储持久化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .enums import EnergyTier, GroupState, ReservationState, SiteStatus, TaskState


@dataclass
class Site:
    """算力站点（园区）。"""

    site_id: str
    region: str
    gpu_capacity: int                 # 每时间槽可用 GPU 数
    power_capacity_kw: float          # 每时间槽园区功耗上限
    kw_per_gpu: float                 # 单 GPU 满载功耗
    energy_tier: EnergyTier           # 园区能耗档位
    cost_per_gpu_slot: float          # 每 GPU·槽的额度单价
    status: SiteStatus = SiteStatus.ONLINE

    def power_for(self, gpus: int) -> float:
        return gpus * self.kw_per_gpu

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "region": self.region,
            "gpu_capacity": self.gpu_capacity,
            "power_capacity_kw": self.power_capacity_kw,
            "kw_per_gpu": self.kw_per_gpu,
            "energy_tier": self.energy_tier.value,
            "cost_per_gpu_slot": self.cost_per_gpu_slot,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Site":
        return cls(
            site_id=data["site_id"],
            region=data["region"],
            gpu_capacity=int(data["gpu_capacity"]),
            power_capacity_kw=float(data["power_capacity_kw"]),
            kw_per_gpu=float(data["kw_per_gpu"]),
            energy_tier=EnergyTier(data["energy_tier"]),
            cost_per_gpu_slot=float(data["cost_per_gpu_slot"]),
            status=SiteStatus(data.get("status", "online")),
        )


@dataclass
class Link:
    """站点间网络链路（双向，带宽按每槽计）。"""

    link_id: str
    site_a: str
    site_b: str
    bandwidth_gbps: float             # 每槽可分配带宽
    cost_per_gb: float                # 每 GB 传输的额度单价

    def other_end(self, site_id: str) -> str:
        if site_id == self.site_a:
            return self.site_b
        if site_id == self.site_b:
            return self.site_a
        raise ValueError(f"链路 {self.link_id} 不包含站点 {site_id}")

    def to_dict(self) -> dict:
        return {
            "link_id": self.link_id,
            "site_a": self.site_a,
            "site_b": self.site_b,
            "bandwidth_gbps": self.bandwidth_gbps,
            "cost_per_gb": self.cost_per_gb,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Link":
        return cls(
            link_id=data["link_id"],
            site_a=data["site_a"],
            site_b=data["site_b"],
            bandwidth_gbps=float(data["bandwidth_gbps"]),
            cost_per_gb=float(data["cost_per_gb"]),
        )


@dataclass
class Dataset:
    """受驻留约束的数据副本集合。

    ``residency_regions`` 为允许使用该数据执行计算的区域白名单；为空表示
    不限制。数据主副本存放于 ``site_id``，其余副本在 ``replica_sites``。
    """

    dataset_id: str
    site_id: str
    size_gb: float
    residency_regions: frozenset[str] = frozenset()
    replica_sites: frozenset[str] = frozenset()

    @property
    def locations(self) -> frozenset[str]:
        return self.replica_sites | {self.site_id}

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "site_id": self.site_id,
            "size_gb": self.size_gb,
            "residency_regions": sorted(self.residency_regions),
            "replica_sites": sorted(self.replica_sites),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Dataset":
        return cls(
            dataset_id=data["dataset_id"],
            site_id=data["site_id"],
            size_gb=float(data["size_gb"]),
            residency_regions=frozenset(data.get("residency_regions", [])),
            replica_sites=frozenset(data.get("replica_sites", [])),
        )


@dataclass
class Tenant:
    """租户额度账户。"""

    tenant_id: str
    quota_limit: float                 # 累计额度上限

    def to_dict(self) -> dict:
        return {"tenant_id": self.tenant_id, "quota_limit": self.quota_limit}

    @classmethod
    def from_dict(cls, data: dict) -> "Tenant":
        return cls(tenant_id=data["tenant_id"], quota_limit=float(data["quota_limit"]))


@dataclass
class TaskSpec:
    """子任务规格（提交时声明，不可变）。"""

    task_id: str
    gpus: int
    duration_slots: int
    inputs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    earliest_start: int = 0
    deadline: int | None = None        # 必须完成的绝对时间槽

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "gpus": self.gpus,
            "duration_slots": self.duration_slots,
            "inputs": list(self.inputs),
            "depends_on": list(self.depends_on),
            "residency_regions": [],
            "earliest_start": self.earliest_start,
            "deadline": self.deadline,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        return cls(
            task_id=data["task_id"],
            gpus=int(data["gpus"]),
            duration_slots=int(data["duration_slots"]),
            inputs=tuple(data.get("inputs", [])),
            depends_on=tuple(data.get("depends_on", [])),
            earliest_start=int(data.get("earliest_start", 0)),
            deadline=data.get("deadline"),
        )


@dataclass
class GroupSpec:
    """一次提交的带依赖作业组规格。"""

    group_id: str
    tenant_id: str
    tasks: list[TaskSpec]
    reservation_ttl_slots: int = 2
    transfer_budget_gb: float | None = None   # 作业组跨网传输总量预算（GB·跳）

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "tenant_id": self.tenant_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "reservation_ttl_slots": self.reservation_ttl_slots,
            "transfer_budget_gb": self.transfer_budget_gb,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroupSpec":
        return cls(
            group_id=data["group_id"],
            tenant_id=data["tenant_id"],
            tasks=[TaskSpec.from_dict(t) for t in data["tasks"]],
            reservation_ttl_slots=int(data.get("reservation_ttl_slots", 2)),
            transfer_budget_gb=data.get("transfer_budget_gb"),
        )


@dataclass
class TransferPlan:
    """单个输入数据集的传输计划：沿 ``edge_ids`` 在固定槽段以固定速率占用。"""

    dataset_id: str
    source_site: str
    edge_ids: list[str]
    size_gb: float
    slots: int
    rate_gbps: float
    tx_start: int
    tx_finish: int

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "source_site": self.source_site,
            "edge_ids": list(self.edge_ids),
            "size_gb": self.size_gb,
            "slots": self.slots,
            "rate_gbps": self.rate_gbps,
            "tx_start": self.tx_start,
            "tx_finish": self.tx_finish,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TransferPlan":
        return cls(
            dataset_id=data["dataset_id"],
            source_site=data["source_site"],
            edge_ids=list(data["edge_ids"]),
            size_gb=float(data["size_gb"]),
            slots=int(data["slots"]),
            rate_gbps=float(data["rate_gbps"]),
            tx_start=int(data["tx_start"]),
            tx_finish=int(data["tx_finish"]),
        )


@dataclass
class Placement:
    """单个子任务的安置结论（规划产物）。"""

    task_id: str
    site_id: str
    start_slot: int
    finish_slot: int
    attempt: int
    transfers: list[TransferPlan] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)  # dataset_id -> 源站点
    quota_cost: float = 0.0
    energy_score: float = 0.0
    transfer_cost: float = 0.0
    network_cost: float = 0.0
    slack: int = 0
    score_total: float = 0.0
    migrated_from: str | None = None

    def transfer_cost_gb(self) -> float:
        """跨网传输总量（GB·跳），用于传输预算硬约束。"""
        return sum(t.size_gb * len(t.edge_ids) for t in self.transfers)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "site_id": self.site_id,
            "start_slot": self.start_slot,
            "finish_slot": self.finish_slot,
            "attempt": self.attempt,
            "transfers": [t.to_dict() for t in self.transfers],
            "sources": dict(self.sources),
            "quota_cost": self.quota_cost,
            "energy_score": self.energy_score,
            "transfer_cost": self.transfer_cost,
            "network_cost": self.network_cost,
            "slack": self.slack,
            "score_total": self.score_total,
            "migrated_from": self.migrated_from,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Placement":
        return cls(
            task_id=data["task_id"],
            site_id=data["site_id"],
            start_slot=int(data["start_slot"]),
            finish_slot=int(data["finish_slot"]),
            attempt=int(data.get("attempt", 1)),
            transfers=[TransferPlan.from_dict(t) for t in data.get("transfers", [])],
            sources=dict(data.get("sources", {})),
            quota_cost=float(data.get("quota_cost", 0.0)),
            energy_score=float(data.get("energy_score", 0.0)),
            transfer_cost=float(data.get("transfer_cost", 0.0)),
            network_cost=float(data.get("network_cost", 0.0)),
            slack=int(data.get("slack", 0)),
            score_total=float(data.get("score_total", 0.0)),
            migrated_from=data.get("migrated_from"),
        )


@dataclass
class TaskRuntime:
    """子任务运行时状态。"""

    spec: TaskSpec
    state: TaskState = TaskState.WAITING
    placement: Placement | None = None
    attempt: int = 1
    run_progress: int = 0
    fail_permanent: bool = False     # 由故障注入置位：执行失败，不可迁移
    last_event: str = ""

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "state": self.state.value,
            "placement": self.placement.to_dict() if self.placement else None,
            "attempt": self.attempt,
            "run_progress": self.run_progress,
            "fail_permanent": self.fail_permanent,
            "last_event": self.last_event,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskRuntime":
        return cls(
            spec=TaskSpec.from_dict(data["spec"]),
            state=TaskState(data["state"]),
            placement=Placement.from_dict(data["placement"]) if data.get("placement") else None,
            attempt=int(data.get("attempt", 1)),
            run_progress=int(data.get("run_progress", 0)),
            fail_permanent=bool(data.get("fail_permanent", False)),
            last_event=data.get("last_event", ""),
        )


@dataclass
class ReservationLine:
    """预留中单个子任务的账本行（容量 / 带宽 / 额度都挂在本行上）。"""

    line_id: str
    task_id: str
    attempt: int
    quota_cost: float
    state: ReservationState = ReservationState.HELD

    def to_dict(self) -> dict:
        return {
            "line_id": self.line_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "quota_cost": self.quota_cost,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReservationLine":
        return cls(
            line_id=data["line_id"],
            task_id=data["task_id"],
            attempt=int(data["attempt"]),
            quota_cost=float(data["quota_cost"]),
            state=ReservationState(data["state"]),
        )


@dataclass
class GroupReservation:
    """作业组级预留：若干账本行 + TTL。"""

    reservation_id: str
    group_id: str
    tenant_id: str
    lines: dict[str, ReservationLine]   # task_id -> line
    created_at: int
    expires_at: int
    state: ReservationState = ReservationState.HELD

    def to_dict(self) -> dict:
        return {
            "reservation_id": self.reservation_id,
            "group_id": self.group_id,
            "tenant_id": self.tenant_id,
            "lines": {tid: line.to_dict() for tid, line in self.lines.items()},
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroupReservation":
        return cls(
            reservation_id=data["reservation_id"],
            group_id=data["group_id"],
            tenant_id=data["tenant_id"],
            lines={tid: ReservationLine.from_dict(v) for tid, v in data["lines"].items()},
            created_at=int(data["created_at"]),
            expires_at=int(data["expires_at"]),
            state=ReservationState(data["state"]),
        )


@dataclass
class JobGroup:
    """作业组聚合根。"""

    spec: GroupSpec
    state: GroupState = GroupState.SUBMITTED
    tasks: dict[str, TaskRuntime] = field(default_factory=dict)
    reservation: GroupReservation | None = None
    explanation_ref: str | None = None
    created_at: int = 0
    state_history: list[tuple[int, str, str]] = field(default_factory=list)

    @property
    def group_id(self) -> str:
        return self.spec.group_id

    @property
    def tenant_id(self) -> str:
        return self.spec.tenant_id

    def record(self, now: int, state: GroupState, note: str = "") -> None:
        self.state = state
        self.state_history.append((now, state.value, note))

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "state": self.state.value,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "reservation": self.reservation.to_dict() if self.reservation else None,
            "explanation_ref": self.explanation_ref,
            "created_at": self.created_at,
            "state_history": [[t, s, n] for t, s, n in self.state_history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobGroup":
        group = cls(
            spec=GroupSpec.from_dict(data["spec"]),
            state=GroupState(data["state"]),
            tasks={tid: TaskRuntime.from_dict(v) for tid, v in data["tasks"].items()},
            reservation=(
                GroupReservation.from_dict(data["reservation"])
                if data.get("reservation")
                else None
            ),
            explanation_ref=data.get("explanation_ref"),
            created_at=int(data.get("created_at", 0)),
            state_history=[(int(t), s, n) for t, s, n in data.get("state_history", [])],
        )
        return group
