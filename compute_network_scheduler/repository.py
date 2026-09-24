"""持久化端口与 JSON 文件适配器。

仓储保存拓扑、两类账本、作业组、决策解释与虚拟时钟，服务重启后可用
:meth:`JsonRepository.load` 完整恢复虚拟时间与所有占用。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Protocol

from .clock import VirtualClock
from .explanation import Explanation
from .ledgers import CapacityLedger, QuotaLedger
from .models import JobGroup
from .topology import Topology


class Repository(Protocol):
    def save(self, state: "ServiceState") -> None: ...
    def load(self) -> "ServiceState | None": ...


class ServiceState:
    """服务全部可持久化状态的聚合容器。"""

    def __init__(
        self,
        topology: Topology | None = None,
        capacity: CapacityLedger | None = None,
        quota: QuotaLedger | None = None,
        clock: VirtualClock | None = None,
        groups: dict[str, JobGroup] | None = None,
        decisions: dict[str, Explanation] | None = None,
        decision_index: dict[str, list[str]] | None = None,
        counters: dict[str, int] | None = None,
    ) -> None:
        self.topology = topology or Topology()
        self.capacity = capacity or CapacityLedger()
        self.quota = quota or QuotaLedger()
        self.clock = clock or VirtualClock()
        self.groups: dict[str, JobGroup] = groups or {}
        self.decisions: dict[str, Explanation] = decisions or {}
        # group_id -> [decision_id, ...] （按时间先后）
        self.decision_index: dict[str, list[str]] = decision_index or {}
        self.counters: dict[str, int] = counters or {}

    def to_dict(self) -> dict:
        return {
            "topology": self.topology.to_dict(),
            "capacity": self.capacity.to_dict(),
            "quota": self.quota.to_dict(),
            "clock": self.clock.snapshot(),
            "groups": {gid: g.to_dict() for gid, g in self.groups.items()},
            "decisions": {did: d.to_dict() for did, d in self.decisions.items()},
            "decision_index": dict(self.decision_index),
            "counters": dict(self.counters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ServiceState":
        return cls(
            topology=Topology.from_dict(data["topology"]),
            capacity=CapacityLedger.from_dict(data["capacity"]),
            quota=QuotaLedger.from_dict(data["quota"]),
            clock=VirtualClock.restore(int(data["clock"])),
            groups={gid: JobGroup.from_dict(g) for gid, g in data.get("groups", {}).items()},
            decisions={
                did: Explanation.from_dict(d) for did, d in data.get("decisions", {}).items()
            },
            decision_index={k: list(v) for k, v in data.get("decision_index", {}).items()},
            counters={k: int(v) for k, v in data.get("counters", {}).items()},
        )


class InMemoryRepository:
    """测试默认仓储：不落盘。"""

    def __init__(self) -> None:
        self._state: ServiceState | None = None

    def save(self, state: ServiceState) -> None:
        self._state = state

    def load(self) -> ServiceState | None:
        return self._state


class JsonRepository:
    """原子写入的 JSON 文件仓储。"""

    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, state: ServiceState) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state.to_dict(), fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def load(self) -> ServiceState | None:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as fh:
            return ServiceState.from_dict(json.load(fh))
