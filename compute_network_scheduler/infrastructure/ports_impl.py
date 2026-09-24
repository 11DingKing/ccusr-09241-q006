"""端口的基础设施实现：手动时钟、顺序标识与状态事件出口。"""

from __future__ import annotations

from typing import Any

from ..domain.models import Event, State


class ManualClock:
    """手动推进的虚拟时钟，保证业务过程可确定性复现。"""

    def __init__(self, now: int = 0) -> None:
        self._now = int(now)

    def now(self) -> int:
        return self._now

    def advance(self, seconds: int) -> None:
        self._now += int(seconds)


class SequentialIds:
    """确定性顺序标识生成器，序列保存在状态中以便恢复。"""

    def __init__(self, state: State) -> None:
        self._state = state

    def next(self, prefix: str) -> str:
        seq = self._state.id_seq.get(prefix, 0) + 1
        self._state.id_seq[prefix] = seq
        return f"{prefix}-{seq:04d}"


class StateEventSink:
    """将领域事件追加到状态中的事件日志。"""

    def __init__(self, state: State, clock: ManualClock) -> None:
        self._state = state
        self._clock = clock

    def emit(self, kind: str, message: str, data: dict[str, Any]) -> None:
        seq = len(self._state.events) + 1
        self._state.events.append(
            Event(seq=seq, at_seconds=self._clock.now(), kind=kind, message=message, data=data)
        )
