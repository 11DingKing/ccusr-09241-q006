"""领域错误。"""

from __future__ import annotations


class DomainError(Exception):
    """业务规则拒绝时抛出，code 供接口层映射为稳定的错误码。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
