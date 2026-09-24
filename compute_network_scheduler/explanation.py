"""决策解释记录。

每次安置规划都生成一份不可变解释：每个子任务 × 候选站点为何被硬约束排除、
可行候选在哪些软目标上得分、最终选中谁；组级约束（额度、传输预算）的
排除原因也单独记录，供运维侧查询。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 硬约束排除码
SITE_FAILED = "SITE_FAILED"                # 站点已失效
DATA_RESIDENCY = "DATA_RESIDENCY"          # 数据驻留：区域不在白名单
DATA_UNREACHABLE = "DATA_UNREACHABLE"      # 数据所在站点无可达网络路径
SITE_GPU_CAPACITY = "SITE_GPU_CAPACITY"    # 标称 GPU 不足
SITE_POWER_CAPACITY = "SITE_POWER_CAPACITY"  # 园区功耗档位折算后不足
WINDOW_NO_FIT = "WINDOW_NO_FIT"            # 时限窗口内找不到同时满足容量/带宽的槽位
DEADLINE = "DEADLINE"                      # 最早可行开始仍晚于完成时限
TENANT_QUOTA = "TENANT_QUOTA"              # 租户剩余额度不足
TRANSFER_BUDGET = "TRANSFER_BUDGET"        # 作业组网络传输总量超预算

HARD_REASONS = (
    SITE_FAILED,
    DATA_RESIDENCY,
    DATA_UNREACHABLE,
    SITE_GPU_CAPACITY,
    SITE_POWER_CAPACITY,
    WINDOW_NO_FIT,
    DEADLINE,
    TENANT_QUOTA,
    TRANSFER_BUDGET,
)


@dataclass
class SoftBreakdown:
    """可行候选的软目标分项（数值越小越优，slack 越大越优）。"""

    energy: float = 0.0
    transfer_gb: float = 0.0
    quota_cost: float = 0.0
    slack: int = 0
    weighted_total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "energy": self.energy,
            "transfer_gb": self.transfer_gb,
            "quota_cost": self.quota_cost,
            "slack": self.slack,
            "weighted_total": self.weighted_total,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SoftBreakdown":
        return cls(
            energy=float(data["energy"]),
            transfer_gb=float(data["transfer_gb"]),
            quota_cost=float(data["quota_cost"]),
            slack=int(data["slack"]),
            weighted_total=float(data["weighted_total"]),
        )


@dataclass
class CandidateReport:
    task_id: str
    site_id: str
    feasible: bool
    chosen: bool = False
    hard_rejections: list[tuple[str, str]] = field(default_factory=list)  # (code, detail)
    soft: SoftBreakdown | None = None
    start_slot: int | None = None
    finish_slot: int | None = None

    def reject(self, code: str, detail: str) -> None:
        pair = (code, detail)
        if pair not in self.hard_rejections:
            self.hard_rejections.append(pair)
        self.feasible = False

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "site_id": self.site_id,
            "feasible": self.feasible,
            "chosen": self.chosen,
            "hard_rejections": [[c, d] for c, d in self.hard_rejections],
            "soft": self.soft.to_dict() if self.soft else None,
            "start_slot": self.start_slot,
            "finish_slot": self.finish_slot,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateReport":
        return cls(
            task_id=data["task_id"],
            site_id=data["site_id"],
            feasible=bool(data["feasible"]),
            chosen=bool(data["chosen"]),
            hard_rejections=[(c, d) for c, d in data["hard_rejections"]],
            soft=SoftBreakdown.from_dict(data["soft"]) if data.get("soft") else None,
            start_slot=data.get("start_slot"),
            finish_slot=data.get("finish_slot"),
        )


@dataclass
class Explanation:
    """一次规划决策的完整解释。"""

    decision_id: str
    group_id: str
    created_at: int
    kind: str = "initial"   # initial | migration | replan
    candidates: list[CandidateReport] = field(default_factory=list)
    group_rejections: list[tuple[str, str]] = field(default_factory=list)
    soft_weights: dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "group_id": self.group_id,
            "created_at": self.created_at,
            "kind": self.kind,
            "candidates": [c.to_dict() for c in self.candidates],
            "group_rejections": [[c, d] for c, d in self.group_rejections],
            "soft_weights": dict(self.soft_weights),
            "feasible": self.feasible,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Explanation":
        return cls(
            decision_id=data["decision_id"],
            group_id=data["group_id"],
            created_at=int(data["created_at"]),
            kind=data.get("kind", "initial"),
            candidates=[CandidateReport.from_dict(c) for c in data["candidates"]],
            group_rejections=[(c, d) for c, d in data.get("group_rejections", [])],
            soft_weights=dict(data.get("soft_weights", {})),
            feasible=bool(data["feasible"]),
            note=data.get("note", ""),
        )

    # ---- 查询视图 ------------------------------------------------------
    def tasks_without_feasible_candidate(self) -> list[str]:
        blocked: dict[str, bool] = {}
        for cand in self.candidates:
            if cand.feasible:
                blocked[cand.task_id] = False
            else:
                blocked.setdefault(cand.task_id, True)
        return [tid for tid, no_way in blocked.items() if no_way]
