"""资源用量推导与硬约束评估。

核心原则：算力时段、园区能耗、路径带宽、租户并发额度与传输预算的占用
全部从处于持有状态的预留记录推导，不存在独立的计数器。因此任何操作的
重试只要复用或替换预留记录，就不会造成重复扣减。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .models import (
    RESERVATION_HOLDING_STATES,
    Dataset,
    Job,
    Link,
    Reservation,
    ReservationStatus,
    Site,
    SiteStatus,
    State,
    Task,
    Tenant,
    TransferPlan,
)

# 硬约束代码（用于决策报告）
SITE_DOWN = "SITE_DOWN"
DATA_RESIDENCY = "DATA_RESIDENCY"
NO_PATH = "NO_PATH"
TRANSFER_BUDGET = "TRANSFER_BUDGET"
TENANT_QUOTA = "TENANT_QUOTA"
COMPUTE_CAPACITY = "COMPUTE_CAPACITY"
ENERGY_CAP = "ENERGY_CAP"
LINK_BANDWIDTH = "LINK_BANDWIDTH"
DEADLINE = "DEADLINE"


class UsageView:
    """某一时刻全量资源占用的快照，由持有中的预留推导。"""

    def __init__(self, state: State) -> None:
        # site_id -> slot -> 已占用算力
        self.compute: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # park_id -> slot -> 已占用能耗
        self.energy: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        # link_id -> slot -> 并发传输数
        self.link: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # tenant_id -> slot -> 已占用并发额度
        self.tenant: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for r in state.reservations.values():
            if r.status not in RESERVATION_HOLDING_STATES:
                continue
            site = state.sites.get(r.site_id)
            for s in range(r.start_slot, r.end_slot):
                self.compute[r.site_id][s] += r.compute_units
                if site is not None:
                    self.energy[site.park_id][s] += r.energy_per_slot
                self.tenant[r.tenant_id][s] += r.compute_units
            if r.transfer is not None:
                for s in range(r.transfer.start_slot, r.transfer.end_slot):
                    for lid in r.transfer.path_link_ids:
                        self.link[lid][s] += 1


def tenant_transfer_usage(state: State, tenant_id: str) -> tuple[float, float]:
    """返回（已实际扣减的传输量，预留暂扣的传输量）。

    确认时传输预算转为实际扣减且不可逆；预留持有期间仅暂扣；
    预留过期或取消（未确认）不占用预算。
    """
    consumed = 0.0
    held = 0.0
    for r in state.reservations.values():
        if r.tenant_id != tenant_id or r.transfer is None:
            continue
        if r.transfer_charged:
            consumed += r.transfer.size_gb
        elif r.status == ReservationStatus.HELD:
            held += r.transfer.size_gb
    return consumed, held


def find_path(state: State, src: str, dst: str) -> Optional[list[str]]:
    """计算两点间的最宽路径（最大化路径最小带宽）。

    平局时依次取跳数更少、字典序更小的路径，保证确定性。
    途经站点与目的站点必须为 UP；源站点允许为 DOWN——站点失效视为
    算力失效，其数据仍可从持久存储读出，以支持受控迁移。
    返回链路 id 列表；同站点返回空列表。
    """
    if src == dst:
        return []
    if src not in state.sites or dst not in state.sites:
        return None
    adjacency: dict[str, list[tuple[str, Link]]] = defaultdict(list)
    for link in state.links.values():
        adjacency[link.site_a].append((link.site_b, link))
        adjacency[link.site_b].append((link.site_a, link))
    for site_id in adjacency:
        adjacency[site_id].sort(key=lambda x: (x[1].id,))

    # best[site] = (min_bw, hops, path)
    best: dict[str, tuple[float, int, tuple[str, ...]]] = {src: (math.inf, 0, ())}
    # 拓扑规模小，使用松弛迭代直至收敛（Bellman-Ford 风格的最大化最小值）
    changed = True
    while changed:
        changed = False
        for site_id in sorted(list(best.keys())):
            bw_s, hops_s, path_s = best[site_id]
            for nxt, link in adjacency.get(site_id, []):
                if state.sites[nxt].status != SiteStatus.UP:
                    continue
                cand = (min(bw_s, link.bandwidth_gb_per_s), hops_s + 1, path_s + (link.id,))
                cur = best.get(nxt)
                if cur is None or _path_better(cand, cur):
                    best[nxt] = cand
                    changed = True
    if dst not in best:
        return None
    return list(best[dst][2])


def _path_better(
    cand: tuple[float, int, tuple[str, ...]], cur: tuple[float, int, tuple[str, ...]]
) -> bool:
    """带宽更大者优先；其次跳数更少者；再次链路 id 字典序更小者。"""
    if cand[0] != cur[0]:
        return cand[0] > cur[0]
    if cand[1] != cur[1]:
        return cand[1] < cur[1]
    return cand[2] < cur[2]


@dataclass
class PlacementPlan:
    site_id: str
    start_slot: int
    end_slot: int
    energy_per_slot: float
    transfer: Optional[TransferPlan]


def transfer_slots_for(size_gb: float, path_link_ids: list[str], state: State) -> int:
    """按路径最窄链路计算传输所需时段数。"""
    if not path_link_ids:
        return 0
    min_bw = min(state.links[lid].bandwidth_gb_per_s for lid in path_link_ids)
    per_slot = min_bw * state.settings.slot_seconds
    return max(1, math.ceil(size_gb / per_slot))


def evaluate_site(
    state: State,
    usage: UsageView,
    job: Job,
    task: Task,
    tenant: Tenant,
    dataset: Dataset,
    site: Site,
) -> tuple[list[str], Optional[PlacementPlan]]:
    """评估单个候选站点，返回（硬约束违反列表, 可行计划）。

    违反列表为空时计划一定存在；计划为 None 时违反列表一定非空。
    """
    violations: list[str] = []

    if site.status != SiteStatus.UP:
        violations.append(SITE_DOWN)
    if site.id not in dataset.allowed_site_ids:
        violations.append(DATA_RESIDENCY)

    # 传输计划（数据不在本地时）
    path: Optional[list[str]] = None
    t_slots = 0
    need_transfer = dataset.location_site_id != site.id
    if need_transfer:
        path = find_path(state, dataset.location_site_id, site.id)
        if path is None:
            violations.append(NO_PATH)
        else:
            t_slots = transfer_slots_for(task.shard_gb, path, state)
        consumed, held = tenant_transfer_usage(state, job.tenant_id)
        if consumed + held + task.shard_gb > tenant.transfer_budget_gb + 1e-9:
            violations.append(TRANSFER_BUDGET)

    if violations:
        return violations, None

    energy_per_slot = task.compute_units * site.kwh_per_unit
    park = state.parks[site.park_id]
    now_slot = state.now_slot

    earliest = max(now_slot, job.earliest_start_slot) + t_slots
    latest_start = job.deadline_slot - task.duration_slots
    if earliest > latest_start:
        return [DEADLINE], None

    window_failures: set[str] = set()
    for start in range(earliest, latest_start + 1):
        failed = _window_violations(
            state, usage, job, task, tenant, site, park.energy_cap_per_slot,
            energy_per_slot, start, path, t_slots,
        )
        if not failed:
            transfer = None
            if need_transfer and path is not None:
                transfer = TransferPlan(
                    from_site=dataset.location_site_id,
                    to_site=site.id,
                    path_link_ids=path,
                    size_gb=task.shard_gb,
                    start_slot=start - t_slots,
                    duration_slots=t_slots,
                )
            return [], PlacementPlan(
                site_id=site.id,
                start_slot=start,
                end_slot=start + task.duration_slots,
                energy_per_slot=energy_per_slot,
                transfer=transfer,
            )
        window_failures.update(failed)
    return sorted(window_failures), None


def _window_violations(
    state: State,
    usage: UsageView,
    job: Job,
    task: Task,
    tenant: Tenant,
    site: Site,
    energy_cap: float,
    energy_per_slot: float,
    start: int,
    path: Optional[list[str]],
    t_slots: int,
) -> set[str]:
    failed: set[str] = set()
    for s in range(start, start + task.duration_slots):
        if usage.compute[site.id][s] + task.compute_units > site.compute_units_per_slot:
            failed.add(COMPUTE_CAPACITY)
        if usage.energy[site.park_id][s] + energy_per_slot > energy_cap + 1e-9:
            failed.add(ENERGY_CAP)
        if usage.tenant[job.tenant_id][s] + task.compute_units > tenant.max_concurrent_units:
            failed.add(TENANT_QUOTA)
    if path:
        for s in range(start - t_slots, start):
            for lid in path:
                if usage.link[lid][s] + 1 > state.links[lid].max_concurrent_transfers:
                    failed.add(LINK_BANDWIDTH)
    return failed
