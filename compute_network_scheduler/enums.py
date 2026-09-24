"""领域枚举与基础结构。"""

from __future__ import annotations

import enum


class SiteStatus(enum.Enum):
    ONLINE = "online"
    FAILED = "failed"


class GroupState(enum.Enum):
    """作业组生命周期。

    SUBMITTED -> PLANNED（已有可执行安置）
    PLANNED   -> RESERVED（容量已预留，等待确认）
    RESERVED  -> CONFIRMED（已确认，等待执行）
    CONFIRMED -> RUNNING（前置依赖满足，至少一个子任务在执行）
    任意活跃态 -> CANCELLED（执行前取消 / 迁移前取消）
    RUNNING   -> COMPLETED（所有非中止子任务越过屏障汇合）
    任意态     -> FAILED（子任务中止且无法再安置）
    RESERVED  -> PLANNED（TTL 超时，预留释放后回到可规划态）
    """

    SUBMITTED = "submitted"
    PLANNED = "planned"
    RESERVED = "reserved"
    CONFIRMED = "confirmed"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskState(enum.Enum):
    """子任务生命周期。"""

    WAITING = "waiting"        # 尚未获得安置
    RESERVED = "reserved"      # 容量已预留
    CONFIRMED = "confirmed"    # 已确认，等待依赖满足 / 执行窗口
    READY = "ready"            # 依赖已满足，可开始执行
    RUNNING = "running"
    SUCCEEDED = "succeeded"    # 执行成功，等待屏障
    ABORTED = "aborted"        # 执行失败 / 所在站点失效
    CANCELLED = "cancelled"
    COMPLETED = "completed"    # 屏障放行后的最终完成态


class ReservationState(enum.Enum):
    HELD = "held"
    CONFIRMED = "confirmed"
    RELEASED = "released"  # 超时 / 取消 / 迁移换绑释放


class EnergyTier(enum.Enum):
    """园区能耗档位：档位越低单位能耗成本越优。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def weight(self) -> float:
        return {"low": 1.0, "medium": 1.6, "high": 2.5}[self.value]
