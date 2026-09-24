"""可替换端口：时间、标识生成与事件观测。

所有用例服务仅依赖这些协议，测试与命令行可注入不同实现，
从而确定性复现业务过程。
"""

from __future__ import annotations

from typing import Any, Protocol


class Clock(Protocol):
    """虚拟时钟，返回当前秒数。"""

    def now(self) -> int: ...

    def advance(self, seconds: int) -> None: ...


class IdGenerator(Protocol):
    """确定性标识生成器。"""

    def next(self, prefix: str) -> str: ...


class EventSink(Protocol):
    """领域事件出口。"""

    def emit(self, kind: str, message: str, data: dict[str, Any]) -> None: ...
