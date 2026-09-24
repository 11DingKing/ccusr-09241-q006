"""安置引擎：对每个候选站点执行硬约束筛选，再按软目标加权排序。

引擎是纯函数式的：只读状态并产出决策报告与计划，不修改任何内容；
预留的创建由应用服务完成。
"""

from __future__ import annotations

from typing import Optional

from .constraints import PlacementPlan, UsageView, evaluate_site
from .models import (
    CandidateReport,
    Dataset,
    DecisionReport,
    Job,
    Site,
    State,
    Task,
    Tenant,
)


def soft_scores(
    state: State,
    usage: UsageView,
    job: Job,
    task: Task,
    site: Site,
    plan: PlacementPlan,
) -> dict[str, float]:
    """计算参与排序的软目标得分，各项均在 [0, 1]，越大越优。"""
    # 数据局部性：无需传输最优；传输越久越差
    if plan.transfer is None:
        locality = 1.0
    else:
        locality = 1.0 / (1.0 + plan.transfer.duration_slots)
    # 能耗效率：档位越低越绿色
    max_tier = max(s.energy_tier for s in state.sites.values())
    energy = (max_tier + 1 - site.energy_tier) / max_tier
    # 完成时间：越早完成越优
    horizon = max(1, job.deadline_slot - state.now_slot)
    completion = max(0.0, (job.deadline_slot - plan.end_slot) / horizon)
    # 容量余量：执行窗口内平均剩余算力占比
    free_ratio = 0.0
    for s in range(plan.start_slot, plan.end_slot):
        used = usage.compute[site.id][s] + task.compute_units
        free_ratio += max(0.0, (site.compute_units_per_slot - used) / site.compute_units_per_slot)
    headroom = free_ratio / task.duration_slots
    return {
        "data_locality": round(locality, 6),
        "energy_efficiency": round(energy, 6),
        "completion_time": round(completion, 6),
        "capacity_headroom": round(headroom, 6),
    }


def evaluate_task(
    state: State,
    job: Job,
    task: Task,
    report_id: str,
) -> tuple[DecisionReport, Optional[PlacementPlan]]:
    """评估全部候选站点，返回决策报告与最优可行计划。"""
    tenant: Tenant = state.tenants[job.tenant_id]
    dataset: Dataset = state.datasets[job.dataset_id]
    usage = UsageView(state)
    weights = state.settings.soft_weights

    candidates: list[CandidateReport] = []
    plans: dict[str, PlacementPlan] = {}
    for site in sorted(state.sites.values(), key=lambda s: s.id):
        violations, plan = evaluate_site(state, usage, job, task, tenant, dataset, site)
        if plan is None:
            candidates.append(
                CandidateReport(site_id=site.id, feasible=False, hard_violations=violations)
            )
            continue
        scores = soft_scores(state, usage, job, task, site, plan)
        total = round(sum(weights.get(k, 0.0) * v for k, v in scores.items()), 6)
        candidates.append(
            CandidateReport(
                site_id=site.id,
                feasible=True,
                soft_scores=scores,
                total_score=total,
                planned_start_slot=plan.start_slot,
                planned_end_slot=plan.end_slot,
            )
        )
        plans[site.id] = plan

    feasible = [c for c in candidates if c.feasible]
    chosen: Optional[CandidateReport] = None
    if feasible:
        # 总分降序，平局按站点 id 升序，保证确定性
        chosen = sorted(feasible, key=lambda c: (-(c.total_score or 0.0), c.site_id))[0]
    # 报告内候选排序：可行者按得分降序在前，不可行者按站点 id 在后
    candidates.sort(
        key=lambda c: (0 if c.feasible else 1, -(c.total_score or 0.0), c.site_id)
    )
    chosen_plan = plans.get(chosen.site_id) if chosen else None
    report = DecisionReport(
        id=report_id,
        task_id=task.id,
        job_id=job.id,
        created_at_slot=state.now_slot,
        candidates=candidates,
        chosen_site_id=chosen.site_id if chosen else None,
        reservation_id=None,
        attempt=task.attempt,
        note="" if chosen else "无可行站点",
    )
    return report, chosen_plan
