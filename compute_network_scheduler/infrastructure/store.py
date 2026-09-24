"""状态持久化：JSON 文件存储，原子写入。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..domain.models import State


class JsonStateStore:
    """将全量状态保存为单个 JSON 文件。

    运行数据默认存放于用户目录（见 interface.cli 默认路径），
    不写入源码目录。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> State:
        if not self.path.exists():
            return State()
        with self.path.open("r", encoding="utf-8") as f:
            return State.from_dict(json.load(f))

    def save(self, state: State) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)
