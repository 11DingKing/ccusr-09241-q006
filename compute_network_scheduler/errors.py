"""领域异常类型。"""

from __future__ import annotations


class SchedulerError(Exception):
    """所有调度领域错误的基类。"""


class ValidationError(SchedulerError):
    """提交的作业组或资源配置不合法。"""


class StateConflictError(SchedulerError):
    """操作与对象当前状态冲突（如重复确认、执行后取消）。"""


class PlacementImpossibleError(SchedulerError):
    """没有任何安置方案能满足全部硬约束。

    :ivar exclusions: 每个被排除候选的结构化原因，供决策解释查询。
    """

    def __init__(self, message: str, exclusions: list | None = None) -> None:
        super().__init__(message)
        self.exclusions = exclusions or []


class ReservationExpiredError(StateConflictError):
    """预留已超过 TTL 被自动释放，不能再确认。"""


class NotFoundError(SchedulerError):
    """指定的作业组、子任务或预留不存在。"""


class UnknownSiteError(SchedulerError):
    """引用了未注册的站点。"""


class SiteFailedError(SchedulerError):
    """目标站点已失效。"""
