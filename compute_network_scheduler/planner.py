"""安置规划器。

对作业组按依赖拓扑顺序逐任务规划：对每个在线站点依次施加硬约束
（站点状态、数据驻留、数据可达、GPU/功耗容量、最早开始与完成时限、
租户额度、传输预算），记录排除原因；对可行候选计算软目标分项
（能耗、跨网传输量、额度成本、时限松弛度），按权重排序选优。

规划过程中产生的容量占用先落在临时 :class:`PlanningContext` 上，整组
规划成功后才由应用服务写入正式账本——保证一次决策的原子性。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import explanation as expl
from .clock import VirtualClock
from .enums import SiteStatus
from .ledgers import CapacityLedger, QuotaLedger
from .models import JobGroup, Placement, Site, TaskSpec, TransferPlan
from .topology import Topology

DEFAULT_HORIZON_SLOTS = 64

# 软目标默认权重（数值越小越优；slack 经反向归一后参与）
DEFAULT_WEIGHTS: dict[str, float] = {
    "energy": 0.35,
    "transfer": 0.20,
    "quota": 0.30,
    "slack": 0.15,
}


@dataclass
class PlanResult:
    placements: dict[str, Placement]
    explanation: expl.Explanation
    quota_needed: float
    transfer_total_gb: float


@dataclass
class PlanningContext:
    """单次规划的临时容量视图（正式账本 + 本规划已落定占用）。"""

    base: CapacityLedger
    now: int
    # (site_id, slot) -> {owner: gpus}
    gpu_holds: dict[tuple[str, int], dict[str, int]] = field(default_factory=dict)
    # (link_id, slot) -> {owner: gbps}
    bw_holds: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)
    spent_quota: float = 0.0
    spent_transfer: float = 0.0

    def gpu_free(self, site: Site, slot: int) -> int:
        cap = CapacityLedger.effective_gpu_capacity(site)
        used_base = self.base.gpu_used(site.site_id, slot)
        used_plan = sum(self.gpu_holds.get((site.site_id, slot), {}).values())
        return cap - used_base - used_plan

    def bw_free(self, link_id: str, capacity: float, slot: int) -> float:
        used_base = self.base.bw_used(link_id, slot)
        used_plan = sum(self.bw_holds.get((link_id, slot), {}).values())
        return capacity - used_base - used_plan

    def commit_placement(self, owner: str, spec: TaskSpec, p: Placement) -> None:
        for slot in range(p.start_slot, p.finish_slot):
            self.gpu_holds.setdefault((p.site_id, slot), {})[owner] = spec.gpus
        for tx in p.transfers:
            for link_id in tx.edge_ids:
                for slot in range(tx.tx_start, tx.tx_finish):
                    bucket = self.bw_holds.setdefault((link_id, slot), {})
                    bucket[owner] = bucket.get(owner, 0.0) + tx.rate_gbps

    def apply_to(self, ledger: CapacityLedger) -> None:
        """规划成功后把全部临时占用写入正式账本。"""
        for (site_id, slot), bucket in self.gpu_holds.items():
            for owner, gpus in bucket.items():
                ledger.hold_gpu(site_id, slot, slot + 1, gpus, owner)
        for (link_id, slot), bucket in self.bw_holds.items():
            for owner, rate in bucket.items():
                ledger.hold_bw(link_id, slot, rate, owner)


class Planner:
    def __init__(
        self,
        topology: Topology,
        capacity: CapacityLedger,
        quota: QuotaLedger,
        clock: VirtualClock,
        weights: dict[str, float] | None = None,
        horizon_slots: int = DEFAULT_HORIZON_SLOTS,
    ) -> None:
        self.topo = topology
        self.capacity = capacity
        self.quota = quota
        self.clock = clock
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.horizon = horizon_slots

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def plan_group(
        self,
        group: JobGroup,
        decision_id: str,
        *,
        kind: str = "initial",
        failed_sites: frozenset[str] = frozenset(),
        tasks_to_plan: set[str] | None = None,
    ) -> PlanResult:
        """规划整组或部分子任务（迁移时只规划待重安置集合）。

        未纳入 ``tasks_to_plan`` 的子任务沿用既有安置，其容量占用已在正式
        账本中，临时上下文通过基础账本自然感知；组级额度 / 传输口径则把
        既有支出显式累加进上下文。
        """
        now = self.clock.now
        ctx = PlanningContext(base=self.capacity, now=now)
        explanation = expl.Explanation(
            decision_id=decision_id,
            group_id=group.group_id,
            created_at=now,
            kind=kind,
            soft_weights=dict(self.weights),
        )

        spec_by_id = {t.task_id: t for t in group.spec.tasks}
        order = self._topo_order(group.spec.tasks)
        targets = order if tasks_to_plan is None else [t for t in order if t in tasks_to_plan]

        chosen: dict[str, Placement] = {}
        placements_all: dict[str, Placement] = {}
        if tasks_to_plan is not None:
            for tid, rt in group.tasks.items():
                if tid not in tasks_to_plan and rt.placement is not None:
                    placements_all[tid] = rt.placement
                    ctx.spent_quota += rt.placement.quota_cost
                    ctx.spent_transfer += rt.placement.transfer_cost_gb()

        tenant = self.topo.tenants[group.tenant_id]

        for task_id in targets:
            spec = spec_by_id[task_id]
            attempt = group.tasks[task_id].attempt if task_id in group.tasks else 1
            candidate_reports, feasible_options = self._evaluate_task(
                group, spec, attempt, chosen, ctx, failed_sites
            )
            explanation.candidates.extend(candidate_reports)
            if not feasible_options:
                explanation.feasible = False
                explanation.note = f"子任务 {task_id} 无可行候选站点"
                return PlanResult(
                    placements={}, explanation=explanation,
                    quota_needed=ctx.spent_quota, transfer_total_gb=ctx.spent_transfer,
                )
            pick = self._choose(feasible_options)
            chosen[task_id] = pick
            placements_all[task_id] = pick
            next(r for r in candidate_reports if r.site_id == pick.site_id).chosen = True
            ctx.commit_placement(self._owner(group.group_id, task_id, attempt), spec, pick)
            ctx.spent_quota += pick.quota_cost
            ctx.spent_transfer += pick.transfer_cost_gb()

        # 组级硬约束兜底校验
        available_quota = self.quota.available(tenant.tenant_id, tenant.quota_limit)
        if ctx.spent_quota > available_quota + 1e-9:
            explanation.feasible = False
            explanation.group_rejections.append(
                (expl.TENANT_QUOTA,
                 f"整组最低额度需求 {ctx.spent_quota:.2f} > 租户剩余 {available_quota:.2f}")
            )
        budget = group.spec.transfer_budget_gb
        if budget is not None and ctx.spent_transfer > budget + 1e-9:
            explanation.feasible = False
            explanation.group_rejections.append(
                (expl.TRANSFER_BUDGET,
                 f"整组跨网传输 {ctx.spent_transfer:.1f}GB > 预算 {budget:.1f}GB")
            )

        if not explanation.feasible:
            return PlanResult(
                placements={}, explanation=explanation,
                quota_needed=ctx.spent_quota, transfer_total_gb=ctx.spent_transfer,
            )

        # 选定后把软目标分项写入解释
        for report in explanation.candidates:
            if report.feasible:
                p = chosen.get(report.task_id)
                if p is not None and report.site_id == p.site_id:
                    report.soft.weighted_total = p.score_total
        return PlanResult(
            placements=placements_all,
            explanation=explanation,
            quota_needed=ctx.spent_quota,
            transfer_total_gb=ctx.spent_transfer,
        )

    # ------------------------------------------------------------------
    # 单任务评估
    # ------------------------------------------------------------------
    def _evaluate_task(
        self,
        group: JobGroup,
        spec: TaskSpec,
        attempt: int,
        chosen: dict[str, Placement],
        ctx: PlanningContext,
        failed_sites: frozenset[str],
    ) -> tuple[list[expl.CandidateReport], list[Placement]]:
        reports: list[expl.CandidateReport] = []
        options: list[Placement] = []
        tenant = self.topo.tenants[group.tenant_id]

        dep_finish = 0
        for dep in spec.depends_on:
            dep_p = chosen.get(dep)
            if dep_p is None and dep in group.tasks:
                dep_p = group.tasks[dep].placement
            if dep_p is not None:
                dep_finish = max(dep_finish, dep_p.finish_slot)
        compute_lower = max(self.clock.now, spec.earliest_start, dep_finish)

        for site in self.topo.sites.values():
            report = expl.CandidateReport(
                task_id=spec.task_id, site_id=site.site_id, feasible=True
            )
            hard_block = False

            def reject(code: str, detail: str) -> None:
                nonlocal hard_block
                report.reject(code, detail)
                hard_block = True

            # 1) 站点状态
            if site.status != SiteStatus.ONLINE or site.site_id in failed_sites:
                reject(expl.SITE_FAILED, "站点已失效或处于故障隔离集合")

            # 2) 驻留 + 3) 数据可达（多副本选最宽路径）+ 构造传输规格
            transfer_specs: list[TransferPlan] = []
            for ds_id in spec.inputs:
                dataset = self.topo.datasets[ds_id]
                if dataset.residency_regions and site.region not in dataset.residency_regions:
                    reject(
                        expl.DATA_RESIDENCY,
                        f"数据 {ds_id} 仅允许驻留区域 {sorted(dataset.residency_regions)}，"
                        f"候选站点区域为 {site.region}",
                    )
                    continue
                # 本地副本优先
                if site.site_id in dataset.locations:
                    transfer_specs.append(TransferPlan(
                        dataset_id=ds_id, source_site=site.site_id, edge_ids=[],
                        size_gb=0.0, slots=0, rate_gbps=0.0, tx_start=-1, tx_finish=-1,
                    ))
                    continue
                best_route: tuple[float, int, str, list] | None = None
                for loc in sorted(dataset.locations):
                    found = self.topo.widest_path(
                        loc, site.site_id, failed_sites=failed_sites
                    )
                    if found is None:
                        continue
                    edges, bottleneck = found
                    # 主键：瓶颈带宽最大；并列取跳数最少，再取站点标识决胜
                    key = (bottleneck, -len(edges), loc)
                    if best_route is None or key > (
                        best_route[0], best_route[1], best_route[2]
                    ):
                        best_route = (bottleneck, -len(edges), loc, edges)
                if best_route is None:
                    reject(expl.DATA_UNREACHABLE,
                           f"数据 {ds_id} 的全部副本站点到候选站点均无可达路径")
                    continue
                bottleneck, _neg_hops, source_site, edges = best_route
                slots = max(1, math.ceil(dataset.size_gb * 8 / bottleneck))
                rate = dataset.size_gb * 8 / slots
                transfer_specs.append(TransferPlan(
                    dataset_id=ds_id, source_site=source_site,
                    edge_ids=[e.link_id for e in edges],
                    size_gb=dataset.size_gb, slots=slots, rate_gbps=rate,
                    tx_start=-1, tx_finish=-1,
                ))

            # 4) 标称 GPU / 功耗折算容量
            nominal = site.gpu_capacity
            effective = CapacityLedger.effective_gpu_capacity(site)
            if spec.gpus > nominal:
                reject(expl.SITE_GPU_CAPACITY,
                       f"需求 {spec.gpus} GPU > 站点标称 {nominal}")
            if spec.gpus > effective:
                reject(expl.SITE_POWER_CAPACITY,
                       f"需求 {spec.gpus} GPU > 功耗上限 {site.power_capacity_kw}kW "
                       f"折算等效 {effective}（单机柜 {site.kw_per_gpu}kW）")

            placement: Placement | None = None
            if not hard_block:
                placement = self._find_earliest_window(
                    group, spec, attempt, site, transfer_specs, compute_lower, ctx, reject
                )

            # 5) 租户额度（候选边际口径）
            if placement is not None:
                avail_quota = self.quota.available(tenant.tenant_id, tenant.quota_limit)
                if ctx.spent_quota + placement.quota_cost > avail_quota + 1e-9:
                    reject(expl.TENANT_QUOTA,
                           f"本候选需 {placement.quota_cost:.2f} 额度，"
                           f"组内累计 {ctx.spent_quota:.2f}，租户剩余 {avail_quota:.2f}")
                budget = group.spec.transfer_budget_gb
                if budget is not None and ctx.spent_transfer + placement.transfer_cost_gb() > budget:
                    reject(expl.TRANSFER_BUDGET,
                           f"本候选新增跨网 {placement.transfer_cost_gb():.1f}GB·跳，"
                           f"组内累计 {ctx.spent_transfer:.1f}，预算 {budget:.1f}GB")

            reports.append(report)
            if report.feasible and placement is not None:
                report.start_slot = placement.start_slot
                report.finish_slot = placement.finish_slot
                report.soft = expl.SoftBreakdown(
                    energy=placement.energy_score,
                    transfer_gb=placement.transfer_cost_gb(),
                    quota_cost=placement.quota_cost,
                    slack=placement.slack,
                )
                options.append(placement)

        return reports, options

    def _find_earliest_window(
        self,
        group: JobGroup,
        spec: TaskSpec,
        attempt: int,
        site: Site,
        transfer_specs: list[TransferPlan],
        compute_lower: int,
        ctx: PlanningContext,
        reject,
    ) -> Placement | None:
        """在时限与容量约束下寻找最早可行窗口，失败时记录排除原因。"""
        deadline = spec.deadline
        upper = min(self.horizon, deadline) if deadline is not None else self.horizon
        remote = [t for t in transfer_specs if t.edge_ids]

        for start in range(compute_lower, upper + 1):
            finish = start + spec.duration_slots
            if deadline is not None and finish > deadline:
                break
            if any(ctx.gpu_free(site, s) < spec.gpus for s in range(start, finish)):
                continue
            # 传输调度：在 [now, start) 内为每条输入路径贪心安排最早连续窗口
            scheduled = [TransferPlan(
                dataset_id=t.dataset_id, source_site=t.source_site,
                edge_ids=list(t.edge_ids), size_gb=t.size_gb,
                slots=t.slots, rate_gbps=t.rate_gbps,
                tx_start=-1, tx_finish=-1,
            ) for t in remote]
            bw_view: dict[tuple[str, int], float] = {}
            ok = True
            for tx in sorted(scheduled, key=lambda x: (-x.slots, x.dataset_id)):
                if not self._schedule_transfer(tx, start, ctx, bw_view):
                    ok = False
                    break
            if not ok:
                continue
            return self._build_placement(
                spec, attempt, site, start, finish, transfer_specs, scheduled
            )

        # 区分时限类与容量类排除原因
        if deadline is not None and upper - compute_lower < spec.duration_slots:
            reject(expl.DEADLINE,
                   f"最早开始 {compute_lower} + 时长 {spec.duration_slots} "
                   f"晚于完成时限 {deadline}")
        elif deadline is not None:
            reject(expl.WINDOW_NO_FIT,
                   f"完成时限 {deadline} 前找不到同时满足 GPU/带宽容量的连续窗口")
        else:
            reject(expl.WINDOW_NO_FIT,
                   f"规划视野 {self.horizon} 槽内无满足 GPU/带宽容量的连续窗口")
        return None

    def _schedule_transfer(
        self,
        tx: TransferPlan,
        compute_start: int,
        ctx: PlanningContext,
        bw_view: dict[tuple[str, int], float],
    ) -> bool:
        """在 [now, compute_start) 内为一条传输路径找最早逐边连续窗口。"""
        need = tx.slots
        if need == 0:
            return True
        upper = compute_start - need
        for begin in range(ctx.now, upper + 1):
            fits = True
            for link_id in tx.edge_ids:
                cap = self.topo.links[link_id].bandwidth_gbps
                for slot in range(begin, begin + need):
                    free = ctx.bw_free(link_id, cap, slot) - bw_view.get((link_id, slot), 0.0)
                    if free + 1e-9 < tx.rate_gbps:
                        fits = False
                        break
                if not fits:
                    break
            if fits:
                tx.tx_start = begin
                tx.tx_finish = begin + need
                for link_id in tx.edge_ids:
                    for slot in range(begin, begin + need):
                        key = (link_id, slot)
                        bw_view[key] = bw_view.get(key, 0.0) + tx.rate_gbps
                return True
        return False

    # ------------------------------------------------------------------
    # 计分与选择
    # ------------------------------------------------------------------
    def _build_placement(
        self,
        spec: TaskSpec,
        attempt: int,
        site: Site,
        start: int,
        finish: int,
        transfer_specs: list[TransferPlan],
        scheduled: list[TransferPlan],
    ) -> Placement:
        compute_cost = site.cost_per_gpu_slot * spec.gpus * spec.duration_slots
        energy = spec.gpus * spec.duration_slots * site.energy_tier.weight
        network_cost = 0.0
        transfer_gb_hops = 0.0
        tx_start = None
        tx_finish = None
        sources: dict[str, str] = {}
        for t in transfer_specs:
            sources[t.dataset_id] = t.source_site
        for tx in scheduled:
            tx_start = tx.tx_start if tx_start is None else min(tx_start, tx.tx_start)
            tx_finish = tx.tx_finish if tx_finish is None else max(tx_finish, tx.tx_finish)
            for link_id in tx.edge_ids:
                transfer_gb_hops += tx.size_gb
                network_cost += self.topo.links[link_id].cost_per_gb * tx.size_gb
        return Placement(
            task_id=spec.task_id,
            site_id=site.site_id,
            start_slot=start,
            finish_slot=finish,
            attempt=attempt,
            transfers=scheduled,
            sources=sources,
            quota_cost=round(compute_cost + network_cost, 6),
            energy_score=round(energy, 6),
            transfer_cost=round(transfer_gb_hops, 6),
            network_cost=round(network_cost, 6),
            slack=(spec.deadline - finish) if spec.deadline is not None else 0,
        )

    def _choose(self, options: list[Placement]) -> Placement:
        """软目标 min-max 归一化加权排序；完全并列时按站点标识稳定决胜。"""
        def span(attr: str) -> tuple[float, float]:
            xs = [getattr(o, attr) for o in options]
            return min(xs), max(xs)

        e_lo, e_hi = span("energy_score")
        t_lo, t_hi = span("transfer_cost")
        q_lo, q_hi = span("quota_cost")
        slacks = [o.slack for o in options]
        s_lo, s_hi = min(slacks), max(slacks)

        def norm(v: float, lo: float, hi: float) -> float:
            return 0.0 if hi == lo else (v - lo) / (hi - lo)

        for o in options:
            # slack 越大越优：用 (s_hi - slack) 做反向归一
            slack_rev_lo = 0.0
            slack_rev_hi = max(s_hi - s_lo, 0)
            slack_term = norm(s_hi - o.slack, slack_rev_lo, slack_rev_hi)
            total = (
                self.weights["energy"] * norm(o.energy_score, e_lo, e_hi)
                + self.weights["transfer"] * norm(o.transfer_cost, t_lo, t_hi)
                + self.weights["quota"] * norm(o.quota_cost, q_lo, q_hi)
                + self.weights["slack"] * slack_term
            )
            o.score_total = round(total, 6)
        return min(options, key=lambda o: (o.score_total, o.site_id))

    # ------------------------------------------------------------------
    @staticmethod
    def _owner(group_id: str, task_id: str, attempt: int) -> str:
        return f"{group_id}:{task_id}:a{attempt}"

    @staticmethod
    def _topo_order(tasks: list[TaskSpec]) -> list[str]:
        """Kahn 拓扑排序；依赖缺失或成环直接报错（提交阶段已校验，双保险）。"""
        ids = {t.task_id for t in tasks}
        indeg = {t.task_id: 0 for t in tasks}
        adj: dict[str, list[str]] = {t.task_id: [] for t in tasks}
        for t in tasks:
            for dep in t.depends_on:
                if dep not in ids:
                    raise ValueError(f"任务 {t.task_id} 依赖了不存在的任务 {dep}")
                adj[dep].append(t.task_id)
                indeg[t.task_id] += 1
        ready = sorted(tid for tid, d in indeg.items() if d == 0)
        order: list[str] = []
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            for nxt in sorted(adj[cur]):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
            ready.sort()
        if len(order) != len(tasks):
            raise ValueError("作业组依赖图存在环")
        return order
