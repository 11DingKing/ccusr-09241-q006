"""核心领域模型。

本模块只包含数据结构、状态枚举与序列化逻辑，不执行任何输入输出。
所有实体以字典形式持久化，保证虚拟时间、标识序列与业务状态可以整体
保存、恢复，从而确定性复现任意业务过程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 状态枚举
# ---------------------------------------------------------------------------


class SiteStatus(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class JobState(str, Enum):
    """作业生命周期。

    PENDING   等待依赖完成
    READY     依赖就绪，可参与安置
    RESERVED  已持有预留，等待确认
    SCHEDULED 已确认，等待到达执行时段
    RUNNING   执行中
    SUCCEEDED 汇合条件满足，进入完成态
    FAILED    失败（含汇合条件不可能满足）
    CANCELLED 执行前被取消
    BLOCKED   依赖失败或失效后无可行迁移目标
    """

    PENDING = "PENDING"
    READY = "READY"
    RESERVED = "RESERVED"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"


JOB_TERMINAL_STATES = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.BLOCKED}
# 执行前允许取消的状态集合
JOB_CANCELLABLE_STATES = {JobState.PENDING, JobState.READY, JobState.RESERVED, JobState.SCHEDULED}


class TaskState(str, Enum):
    """子任务生命周期。

    拆分作业的子任务执行完毕后先进入 JOIN_WAIT（等待汇合），
    只有汇合条件满足时才允许进入完成态 SUCCEEDED。
    """

    PENDING = "PENDING"
    RESERVED = "RESERVED"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    JOIN_WAIT = "JOIN_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TASK_ACTIVE_STATES = {TaskState.RESERVED, TaskState.SCHEDULED, TaskState.RUNNING, TaskState.JOIN_WAIT}


class ReservationStatus(str, Enum):
    """预留生命周期。

    HELD       预留持有中（占用算力/带宽/能耗/额度，传输预算仅暂扣）
    CONFIRMED  已确认（传输预算转为实际扣减）
    COMPLETED  关联任务已完成（算力释放，传输扣减保留）
    EXPIRED    超时未确认，自动释放
    CANCELLED  执行前取消或作业终止时释放
    SUPERSEDED 受控迁移后被新预留取代
    """

    HELD = "HELD"
    CONFIRMED = "CONFIRMED"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"


# 占用算力时段 / 能耗档位 / 路径带宽 / 租户并发额度的预留状态
RESERVATION_HOLDING_STATES = {ReservationStatus.HELD, ReservationStatus.CONFIRMED}


# ---------------------------------------------------------------------------
# 静态拓扑与租户
# ---------------------------------------------------------------------------


@dataclass
class Park:
    """园区：能耗上限的作用域。"""

    id: str
    name: str
    energy_cap_per_slot: float  # 每时段能耗上限（千瓦时）

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "energy_cap_per_slot": self.energy_cap_per_slot}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Park":
        return Park(id=d["id"], name=d["name"], energy_cap_per_slot=float(d["energy_cap_per_slot"]))


@dataclass
class Site:
    """算力站点。"""

    id: str
    name: str
    region: str
    park_id: str
    energy_tier: int  # 能耗档位，数值越小越绿色
    compute_units_per_slot: int  # 每时段可用算力
    kwh_per_unit: float  # 每算力单位每时段能耗
    status: SiteStatus = SiteStatus.UP

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "region": self.region,
            "park_id": self.park_id,
            "energy_tier": self.energy_tier,
            "compute_units_per_slot": self.compute_units_per_slot,
            "kwh_per_unit": self.kwh_per_unit,
            "status": self.status.value,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Site":
        return Site(
            id=d["id"],
            name=d["name"],
            region=d["region"],
            park_id=d["park_id"],
            energy_tier=int(d["energy_tier"]),
            compute_units_per_slot=int(d["compute_units_per_slot"]),
            kwh_per_unit=float(d["kwh_per_unit"]),
            status=SiteStatus(d["status"]),
        )


@dataclass
class Link:
    """站点间网络路径（无向）。"""

    id: str
    site_a: str
    site_b: str
    bandwidth_gb_per_s: float
    max_concurrent_transfers: int

    def connects(self, x: str, y: str) -> bool:
        return {self.site_a, self.site_b} == {x, y}

    def other(self, site_id: str) -> Optional[str]:
        if site_id == self.site_a:
            return self.site_b
        if site_id == self.site_b:
            return self.site_a
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "site_a": self.site_a,
            "site_b": self.site_b,
            "bandwidth_gb_per_s": self.bandwidth_gb_per_s,
            "max_concurrent_transfers": self.max_concurrent_transfers,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Link":
        return Link(
            id=d["id"],
            site_a=d["site_a"],
            site_b=d["site_b"],
            bandwidth_gb_per_s=float(d["bandwidth_gb_per_s"]),
            max_concurrent_transfers=int(d["max_concurrent_transfers"]),
        )


@dataclass
class Tenant:
    """租户：并发算力额度与累计网络传输预算。"""

    id: str
    name: str
    max_concurrent_units: int  # 每时段并发算力额度
    transfer_budget_gb: float  # 累计网络传输预算

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "max_concurrent_units": self.max_concurrent_units,
            "transfer_budget_gb": self.transfer_budget_gb,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Tenant":
        return Tenant(
            id=d["id"],
            name=d["name"],
            max_concurrent_units=int(d["max_concurrent_units"]),
            transfer_budget_gb=float(d["transfer_budget_gb"]),
        )


@dataclass
class Dataset:
    """数据集：位置与驻留限制。"""

    id: str
    name: str
    size_gb: float
    location_site_id: str
    allowed_site_ids: list[str]  # 数据驻留限制：仅允许在这些站点处理

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "size_gb": self.size_gb,
            "location_site_id": self.location_site_id,
            "allowed_site_ids": list(self.allowed_site_ids),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Dataset":
        return Dataset(
            id=d["id"],
            name=d["name"],
            size_gb=float(d["size_gb"]),
            location_site_id=d["location_site_id"],
            allowed_site_ids=list(d["allowed_site_ids"]),
        )


# ---------------------------------------------------------------------------
# 作业、子任务与预留
# ---------------------------------------------------------------------------


@dataclass
class JoinSpec:
    """拆分作业的汇合条件。all：全部子任务；quorum：至少 k 个子任务。"""

    kind: str = "all"  # "all" | "quorum"
    k: int = 0

    def required(self, n: int) -> int:
        return n if self.kind == "all" else max(1, min(self.k, n))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "k": self.k}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "JoinSpec":
        return JoinSpec(kind=d.get("kind", "all"), k=int(d.get("k", 0)))


@dataclass
class JobSpec:
    """作业组提交时的单个作业描述。"""

    key: str  # 组内唯一键，依赖关系按 key 引用
    dataset_id: str
    compute_units: int
    duration_slots: int
    deadline_slot: int
    depends_on: list[str] = field(default_factory=list)
    earliest_start_slot: int = 0
    splits: int = 1
    join: JoinSpec = field(default_factory=JoinSpec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dataset_id": self.dataset_id,
            "compute_units": self.compute_units,
            "duration_slots": self.duration_slots,
            "deadline_slot": self.deadline_slot,
            "depends_on": list(self.depends_on),
            "earliest_start_slot": self.earliest_start_slot,
            "splits": self.splits,
            "join": self.join.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "JobSpec":
        return JobSpec(
            key=d["key"],
            dataset_id=d["dataset_id"],
            compute_units=int(d["compute_units"]),
            duration_slots=int(d["duration_slots"]),
            deadline_slot=int(d["deadline_slot"]),
            depends_on=list(d.get("depends_on", [])),
            earliest_start_slot=int(d.get("earliest_start_slot", 0)),
            splits=int(d.get("splits", 1)),
            join=JoinSpec.from_dict(d.get("join", {})),
        )


@dataclass
class Task:
    """子任务：作业的最小执行与安置单元。"""

    id: str
    job_id: str
    seq: int
    compute_units: int
    duration_slots: int
    shard_gb: float  # 该子任务需要传输的数据分片大小
    state: TaskState = TaskState.PENDING
    site_id: Optional[str] = None
    reservation_id: Optional[str] = None
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "seq": self.seq,
            "compute_units": self.compute_units,
            "duration_slots": self.duration_slots,
            "shard_gb": self.shard_gb,
            "state": self.state.value,
            "site_id": self.site_id,
            "reservation_id": self.reservation_id,
            "attempt": self.attempt,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Task":
        return Task(
            id=d["id"],
            job_id=d["job_id"],
            seq=int(d["seq"]),
            compute_units=int(d["compute_units"]),
            duration_slots=int(d["duration_slots"]),
            shard_gb=float(d["shard_gb"]),
            state=TaskState(d["state"]),
            site_id=d.get("site_id"),
            reservation_id=d.get("reservation_id"),
            attempt=int(d.get("attempt", 1)),
        )


@dataclass
class Job:
    id: str
    group_id: str
    tenant_id: str
    key: str
    dataset_id: str
    compute_units: int
    duration_slots: int
    deadline_slot: int
    earliest_start_slot: int
    depends_on: list[str]  # 作业 id 列表
    join: JoinSpec
    state: JobState = JobState.PENDING
    task_ids: list[str] = field(default_factory=list)
    block_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "tenant_id": self.tenant_id,
            "key": self.key,
            "dataset_id": self.dataset_id,
            "compute_units": self.compute_units,
            "duration_slots": self.duration_slots,
            "deadline_slot": self.deadline_slot,
            "earliest_start_slot": self.earliest_start_slot,
            "depends_on": list(self.depends_on),
            "join": self.join.to_dict(),
            "state": self.state.value,
            "task_ids": list(self.task_ids),
            "block_reason": self.block_reason,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Job":
        return Job(
            id=d["id"],
            group_id=d["group_id"],
            tenant_id=d["tenant_id"],
            key=d["key"],
            dataset_id=d["dataset_id"],
            compute_units=int(d["compute_units"]),
            duration_slots=int(d["duration_slots"]),
            deadline_slot=int(d["deadline_slot"]),
            earliest_start_slot=int(d["earliest_start_slot"]),
            depends_on=list(d["depends_on"]),
            join=JoinSpec.from_dict(d["join"]),
            state=JobState(d["state"]),
            task_ids=list(d["task_ids"]),
            block_reason=d.get("block_reason"),
        )


@dataclass
class JobGroup:
    id: str
    tenant_id: str
    submitted_at_slot: int
    job_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "submitted_at_slot": self.submitted_at_slot,
            "job_ids": list(self.job_ids),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "JobGroup":
        return JobGroup(
            id=d["id"],
            tenant_id=d["tenant_id"],
            submitted_at_slot=int(d["submitted_at_slot"]),
            job_ids=list(d["job_ids"]),
        )


@dataclass
class TransferPlan:
    """一次数据分片传输计划。"""

    from_site: str
    to_site: str
    path_link_ids: list[str]
    size_gb: float
    start_slot: int
    duration_slots: int

    @property
    def end_slot(self) -> int:
        return self.start_slot + self.duration_slots

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_site": self.from_site,
            "to_site": self.to_site,
            "path_link_ids": list(self.path_link_ids),
            "size_gb": self.size_gb,
            "start_slot": self.start_slot,
            "duration_slots": self.duration_slots,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TransferPlan":
        return TransferPlan(
            from_site=d["from_site"],
            to_site=d["to_site"],
            path_link_ids=list(d["path_link_ids"]),
            size_gb=float(d["size_gb"]),
            start_slot=int(d["start_slot"]),
            duration_slots=int(d["duration_slots"]),
        )


@dataclass
class Reservation:
    """资源预留。算力时段、能耗、带宽与租户额度的占用全部由预留推导，
    因此同一预留的任何重试都不会重复扣减。"""

    id: str
    task_id: str
    job_id: str
    tenant_id: str
    site_id: str
    start_slot: int
    duration_slots: int
    compute_units: int
    energy_per_slot: float
    transfer: Optional[TransferPlan]
    created_at_seconds: int
    expires_at_seconds: int
    status: ReservationStatus = ReservationStatus.HELD
    transfer_charged: bool = False  # 确认时传输预算转为实际扣减（幂等）

    @property
    def end_slot(self) -> int:
        return self.start_slot + self.duration_slots

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "site_id": self.site_id,
            "start_slot": self.start_slot,
            "duration_slots": self.duration_slots,
            "compute_units": self.compute_units,
            "energy_per_slot": self.energy_per_slot,
            "transfer": self.transfer.to_dict() if self.transfer else None,
            "created_at_seconds": self.created_at_seconds,
            "expires_at_seconds": self.expires_at_seconds,
            "status": self.status.value,
            "transfer_charged": self.transfer_charged,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Reservation":
        return Reservation(
            id=d["id"],
            task_id=d["task_id"],
            job_id=d["job_id"],
            tenant_id=d["tenant_id"],
            site_id=d["site_id"],
            start_slot=int(d["start_slot"]),
            duration_slots=int(d["duration_slots"]),
            compute_units=int(d["compute_units"]),
            energy_per_slot=float(d["energy_per_slot"]),
            transfer=TransferPlan.from_dict(d["transfer"]) if d.get("transfer") else None,
            created_at_seconds=int(d["created_at_seconds"]),
            expires_at_seconds=int(d["expires_at_seconds"]),
            status=ReservationStatus(d["status"]),
            transfer_charged=bool(d.get("transfer_charged", False)),
        )


# ---------------------------------------------------------------------------
# 决策报告（可解释性）
# ---------------------------------------------------------------------------


@dataclass
class CandidateReport:
    """单个候选站点的评估结果。"""

    site_id: str
    feasible: bool
    hard_violations: list[str] = field(default_factory=list)  # 排除该候选的硬约束
    soft_scores: dict[str, float] = field(default_factory=dict)  # 参与排序的软目标得分
    total_score: Optional[float] = None
    planned_start_slot: Optional[int] = None
    planned_end_slot: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "feasible": self.feasible,
            "hard_violations": list(self.hard_violations),
            "soft_scores": dict(self.soft_scores),
            "total_score": self.total_score,
            "planned_start_slot": self.planned_start_slot,
            "planned_end_slot": self.planned_end_slot,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CandidateReport":
        return CandidateReport(
            site_id=d["site_id"],
            feasible=bool(d["feasible"]),
            hard_violations=list(d["hard_violations"]),
            soft_scores=dict(d["soft_scores"]),
            total_score=d.get("total_score"),
            planned_start_slot=d.get("planned_start_slot"),
            planned_end_slot=d.get("planned_end_slot"),
        )


@dataclass
class DecisionReport:
    """一次安置决策的完整记录：哪些硬约束排除了候选，哪些软目标参与了排序。"""

    id: str
    task_id: str
    job_id: str
    created_at_slot: int
    candidates: list[CandidateReport]
    chosen_site_id: Optional[str]
    reservation_id: Optional[str]
    attempt: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "job_id": self.job_id,
            "created_at_slot": self.created_at_slot,
            "candidates": [c.to_dict() for c in self.candidates],
            "chosen_site_id": self.chosen_site_id,
            "reservation_id": self.reservation_id,
            "attempt": self.attempt,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DecisionReport":
        return DecisionReport(
            id=d["id"],
            task_id=d["task_id"],
            job_id=d["job_id"],
            created_at_slot=int(d["created_at_slot"]),
            candidates=[CandidateReport.from_dict(c) for c in d["candidates"]],
            chosen_site_id=d.get("chosen_site_id"),
            reservation_id=d.get("reservation_id"),
            attempt=int(d.get("attempt", 1)),
            note=d.get("note", ""),
        )


# ---------------------------------------------------------------------------
# 事件与全局状态
# ---------------------------------------------------------------------------


@dataclass
class Event:
    seq: int
    at_seconds: int
    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "at_seconds": self.at_seconds,
            "kind": self.kind,
            "message": self.message,
            "data": self.data,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Event":
        return Event(
            seq=int(d["seq"]),
            at_seconds=int(d["at_seconds"]),
            kind=d["kind"],
            message=d["message"],
            data=dict(d.get("data", {})),
        )


@dataclass
class Settings:
    slot_seconds: int = 300
    reservation_ttl_seconds: int = 900
    soft_weights: dict[str, float] = field(
        default_factory=lambda: {
            "data_locality": 3.0,
            "energy_efficiency": 2.0,
            "completion_time": 2.0,
            "capacity_headroom": 1.0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_seconds": self.slot_seconds,
            "reservation_ttl_seconds": self.reservation_ttl_seconds,
            "soft_weights": dict(self.soft_weights),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Settings":
        base = Settings()
        return Settings(
            slot_seconds=int(d.get("slot_seconds", base.slot_seconds)),
            reservation_ttl_seconds=int(d.get("reservation_ttl_seconds", base.reservation_ttl_seconds)),
            soft_weights=dict(d.get("soft_weights", base.soft_weights)),
        )


@dataclass
class State:
    """服务全量状态。时间、标识序列与业务实体集中于此，可整体序列化。"""

    now_seconds: int = 0
    settings: Settings = field(default_factory=Settings)
    id_seq: dict[str, int] = field(default_factory=dict)
    parks: dict[str, Park] = field(default_factory=dict)
    sites: dict[str, Site] = field(default_factory=dict)
    links: dict[str, Link] = field(default_factory=dict)
    tenants: dict[str, Tenant] = field(default_factory=dict)
    datasets: dict[str, Dataset] = field(default_factory=dict)
    groups: dict[str, JobGroup] = field(default_factory=dict)
    jobs: dict[str, Job] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)
    reservations: dict[str, Reservation] = field(default_factory=dict)
    decisions: list[DecisionReport] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    @property
    def now_slot(self) -> int:
        return self.now_seconds // self.settings.slot_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "now_seconds": self.now_seconds,
            "settings": self.settings.to_dict(),
            "id_seq": dict(self.id_seq),
            "parks": {k: v.to_dict() for k, v in self.parks.items()},
            "sites": {k: v.to_dict() for k, v in self.sites.items()},
            "links": {k: v.to_dict() for k, v in self.links.items()},
            "tenants": {k: v.to_dict() for k, v in self.tenants.items()},
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "groups": {k: v.to_dict() for k, v in self.groups.items()},
            "jobs": {k: v.to_dict() for k, v in self.jobs.items()},
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "reservations": {k: v.to_dict() for k, v in self.reservations.items()},
            "decisions": [d.to_dict() for d in self.decisions],
            "events": [e.to_dict() for e in self.events],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "State":
        return State(
            now_seconds=int(d.get("now_seconds", 0)),
            settings=Settings.from_dict(d.get("settings", {})),
            id_seq=dict(d.get("id_seq", {})),
            parks={k: Park.from_dict(v) for k, v in d.get("parks", {}).items()},
            sites={k: Site.from_dict(v) for k, v in d.get("sites", {}).items()},
            links={k: Link.from_dict(v) for k, v in d.get("links", {}).items()},
            tenants={k: Tenant.from_dict(v) for k, v in d.get("tenants", {}).items()},
            datasets={k: Dataset.from_dict(v) for k, v in d.get("datasets", {}).items()},
            groups={k: JobGroup.from_dict(v) for k, v in d.get("groups", {}).items()},
            jobs={k: Job.from_dict(v) for k, v in d.get("jobs", {}).items()},
            tasks={k: Task.from_dict(v) for k, v in d.get("tasks", {}).items()},
            reservations={k: Reservation.from_dict(v) for k, v in d.get("reservations", {}).items()},
            decisions=[DecisionReport.from_dict(x) for x in d.get("decisions", [])],
            events=[Event.from_dict(x) for x in d.get("events", [])],
        )
