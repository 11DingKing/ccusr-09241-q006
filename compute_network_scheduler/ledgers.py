"""时序容量账本与租户额度账本。

- :class:`CapacityLedger` 维护「站点 × 时间槽」的 GPU 占用与「链路 × 时间槽」
  的带宽占用；站点功耗上限折算为等效 GPU 上限参与硬约束。
- :class:`QuotaLedger` 维护租户累计额度，所有扣减 / 退还都带幂等键，
  同一键（``作业组:子任务:尝试序号``）重试不会重复扣减。

账本可序列化为纯字典，随仓储一起持久化。
"""

from __future__ import annotations

from collections import defaultdict

from .models import Site


class CapacityLedger:
    def __init__(self) -> None:
        # site_id -> slot -> {owner_key: gpus}
        self._gpu: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        # link_id -> slot -> {owner_key: gbps}
        self._bw: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    # ---- 基础查询 ------------------------------------------------------
    @staticmethod
    def effective_gpu_capacity(site: Site) -> int:
        """功耗上限折算的等效 GPU 上限与标称 GPU 上限取小。"""
        by_power = int(site.power_capacity_kw // site.kw_per_gpu) if site.kw_per_gpu > 0 else 0
        return min(site.gpu_capacity, by_power)

    def gpu_used(self, site_id: str, slot: int) -> int:
        return sum(self._gpu.get(site_id, {}).get(slot, {}).values())

    def gpu_available(self, site: Site, slot: int) -> int:
        return self.effective_gpu_capacity(site) - self.gpu_used(site.site_id, slot)

    def bw_used(self, link_id: str, slot: int) -> float:
        return sum(self._bw.get(link_id, {}).get(slot, {}).values())

    # ---- GPU -----------------------------------------------------------
    def hold_gpu(self, site_id: str, slot_from: int, slot_to: int, gpus: int, owner: str) -> None:
        """在 [slot_from, slot_to) 上为 ``owner`` 预留 GPU；对同一 owner 幂等。"""
        for slot in range(slot_from, slot_to):
            bucket = self._gpu[site_id][slot]
            if owner in bucket:
                if bucket[owner] != gpus:
                    raise ValueError(f"GPU 占用键 {owner} 与已记录数量冲突")
                continue
            bucket[owner] = gpus

    def release_owner(self, owner: str) -> None:
        """释放某 owner（作业组/子任务尝试）在所有站点、链路上的全部占用。"""
        for site_slots in self._gpu.values():
            for bucket in site_slots.values():
                bucket.pop(owner, None)
        for link_slots in self._bw.values():
            for bucket in link_slots.values():
                bucket.pop(owner, None)

    def release_group(self, group_id: str) -> None:
        """释放整个作业组（含历次尝试）的容量占用。"""
        prefix = group_id + ":"
        for site_slots in self._gpu.values():
            for bucket in site_slots.values():
                for key in [k for k in bucket if k.startswith(prefix)]:
                    del bucket[key]
        for link_slots in self._bw.values():
            for bucket in link_slots.values():
                for key in [k for k in bucket if k.startswith(prefix)]:
                    del bucket[key]

    def can_hold_gpu(self, site: Site, slot_from: int, slot_to: int, gpus: int,
                     owner: str) -> bool:
        cap = self.effective_gpu_capacity(site)
        for slot in range(slot_from, slot_to):
            bucket = self._gpu.get(site.site_id, {}).get(slot, {})
            already = bucket.get(owner, 0)
            used = sum(bucket.values()) - already
            if used + gpus > cap:
                return False
        return True

    # ---- 带宽 ----------------------------------------------------------
    def hold_bw(self, link_id: str, slot: int, gbps: float, owner: str) -> None:
        bucket = self._bw[link_id][slot]
        if owner in bucket:
            if bucket[owner] != gbps:
                raise ValueError(f"带宽占用键 {owner} 与已记录数量冲突")
            return
        bucket[owner] = gbps

    def can_hold_bw(self, link_id: str, capacity_gbps: float, slot: int,
                    gbps: float, owner: str) -> bool:
        bucket = self._bw.get(link_id, {}).get(slot, {})
        already = bucket.get(owner, 0.0)
        used = sum(bucket.values()) - already
        return used + gbps <= capacity_gbps + 1e-9

    # ---- 序列化 --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "gpu": {
                site: {str(slot): bucket for slot, bucket in slots.items()}
                for site, slots in self._gpu.items()
            },
            "bw": {
                link: {str(slot): bucket for slot, bucket in slots.items()}
                for link, slots in self._bw.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CapacityLedger":
        ledger = cls()
        for site, slots in data.get("gpu", {}).items():
            for slot, bucket in slots.items():
                ledger._gpu[site][int(slot)] = dict(bucket)
        for link, slots in data.get("bw", {}).items():
            for slot, bucket in slots.items():
                ledger._bw[link][int(slot)] = {k: float(v) for k, v in bucket.items()}
        return ledger


class QuotaLedger:
    """租户累计额度账本，扣减以幂等键记账，退还只能发生一次。"""

    def __init__(self) -> None:
        # tenant -> {key: {"amount": float, "refunded": bool}}
        self._entries: dict[str, dict[str, dict]] = defaultdict(dict)

    def used(self, tenant_id: str) -> float:
        return sum(
            e["amount"] for e in self._entries.get(tenant_id, {}).values() if not e["refunded"]
        )

    def available(self, tenant_id: str, limit: float) -> float:
        return limit - self.used(tenant_id)

    def has_debit(self, tenant_id: str, key: str) -> bool:
        return key in self._entries.get(tenant_id, {})

    def debit(self, tenant_id: str, key: str, amount: float) -> float:
        """扣减额度。

        - 同一 ``key`` 在未退还时重复调用（请求重试）直接返回既有记录，
          **不重复扣减**；
        - 该键此前已因取消 / TTL 超时 / 迁移换绑被退还时，本调用表示一次
          全新预留，键重新生效（金额需与原记录一致）。
        """
        entries = self._entries[tenant_id]
        if key in entries:
            existing = entries[key]
            if not existing["refunded"]:
                if abs(existing["amount"] - amount) > 1e-9:
                    raise ValueError(f"额度键 {key} 金额冲突：{existing['amount']} != {amount}")
                return existing["amount"]
            # 已退还的键重新生效 = 一次全新预留（安置方案可能已变化）
            existing["amount"] = float(amount)
            existing["refunded"] = False
            return float(amount)
        entries[key] = {"amount": float(amount), "refunded": False}
        return float(amount)

    def refund(self, tenant_id: str, key: str) -> float:
        """退还某键的额度；重复退还返回 0.0，保证幂等。"""
        entry = self._entries.get(tenant_id, {}).get(key)
        if entry is None or entry["refunded"]:
            return 0.0
        entry["refunded"] = True
        return float(entry["amount"])

    def refund_group(self, tenant_id: str, group_id: str) -> float:
        """退还某作业组全部未退键（取消 / 迁移旧绑定时使用）。"""
        prefix = group_id + ":"
        total = 0.0
        for key, entry in self._entries.get(tenant_id, {}).items():
            if key.startswith(prefix) and not entry["refunded"]:
                entry["refunded"] = True
                total += entry["amount"]
        return total

    def to_dict(self) -> dict:
        return {
            tenant: {key: dict(entry) for key, entry in entries.items()}
            for tenant, entries in self._entries.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuotaLedger":
        ledger = cls()
        for tenant, entries in data.items():
            for key, entry in entries.items():
                ledger._entries[tenant][key] = {
                    "amount": float(entry["amount"]),
                    "refunded": bool(entry["refunded"]),
                }
        return ledger
