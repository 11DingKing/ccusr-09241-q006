"""虚拟时间服务。

所有领域决策都以整数时间槽（1 槽 = 1 小时）为单位，时钟只能通过
:meth:`VirtualClock.advance` 显式推进，便于在命令行与测试中确定性复现
预留过期、故障迁移等时间相关过程。
"""

from __future__ import annotations


class VirtualClock:
    """可由外部推进的逻辑时钟，启动时固定在第 0 槽。"""

    SLOT_SECONDS = 3600

    def __init__(self, now: int = 0) -> None:
        if now < 0:
            raise ValueError("虚拟时间不能为负数")
        self._now = now

    @property
    def now(self) -> int:
        return self._now

    def advance(self, slots: int) -> int:
        """把时间向前推进 ``slots`` 个整槽，返回推进后的时刻。"""
        if slots <= 0:
            raise ValueError("推进的时间槽数必须为正整数")
        self._now += slots
        return self._now

    def reset(self, now: int = 0) -> None:
        if now < 0:
            raise ValueError("虚拟时间不能为负数")
        self._now = now

    def snapshot(self) -> int:
        return self._now

    @classmethod
    def restore(cls, value: int) -> "VirtualClock":
        return cls(now=value)
