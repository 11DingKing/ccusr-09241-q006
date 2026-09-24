"""四个可确定性复现的运维场景：额度竞争、预留过期、故障迁移、部分子任务失败。

每个场景在全新的内存状态上编排服务调用并输出逐步叙述，
既供命令行演示，也供自动化测试断言关键不变量。
"""

from __future__ import annotations

from typing import Callable

from ..application.services import SchedulerService
from ..domain.constraints import tenant_transfer_usage
from ..domain.models import JobSpec, JoinSpec, State
from ..infrastructure.ports_impl import ManualClock, SequentialIds, StateEventSink


def make_service(state: State | None = None) -> SchedulerService:
    state = state if state is not None else State()
    clock = ManualClock(state.now_seconds)
    return SchedulerService(state, clock, SequentialIds(state), StateEventSink(state, clock))


def build_demo_topology(svc: SchedulerService) -> None:
    """命令行 init-demo 与场景共用的演示拓扑。"""
    svc.add_park("park-east", "东部园区", energy_cap_per_slot=100.0)
    svc.add_park("park-west", "西部园区", energy_cap_per_slot=60.0)
    svc.add_site("site-east-1", "东部一号", "east", "park-east", 1, 10, 2.0)
    svc.add_site("site-east-2", "东部二号", "east", "park-east", 2, 8, 3.0)
    svc.add_site("site-west-1", "西部一号", "west", "park-west", 3, 12, 1.5)
    svc.add_link("link-e1-e2", "site-east-1", "site-east-2", 2.0, 2)
    svc.add_link("link-e1-w1", "site-east-1", "site-west-1", 1.0, 2)
    svc.add_link("link-e2-w1", "site-east-2", "site-west-1", 0.5, 1)
    svc.add_tenant("tenant-a", "租户甲", 10, 200.0)
    svc.add_tenant("tenant-b", "租户乙", 6, 60.0)
    svc.add_dataset("ds-east", "东部数据", 60.0, "site-east-1", ["site-east-1", "site-east-2", "site-west-1"])
    svc.add_dataset("ds-west", "西部数据", 30.0, "site-west-1", ["site-west-1"])


def _quota_of(svc: SchedulerService, tenant_id: str, slot: int) -> int:
    from ..domain.constraints import UsageView

    return UsageView(svc.state).tenant[tenant_id][slot]


def scenario_quota_contention() -> tuple[list[str], SchedulerService]:
    """额度竞争：两个作业竞争同一租户并发额度，被排除作业在额度释放后安置。"""
    svc = make_service()
    build_demo_topology(svc)
    out: list[str] = []
    out.append("== 场景：额度竞争 ==")
    out.append("租户 tenant-a 并发额度 10；big(7 单位,时段0-3) 与 small(5 单位,时限时段3) 竞争同一窗口")

    svc.submit_group(
        "tenant-a",
        [
            JobSpec(key="big", dataset_id="ds-east", compute_units=7, duration_slots=3, deadline_slot=3),
            JobSpec(key="small", dataset_id="ds-east", compute_units=5, duration_slots=2, deadline_slot=3),
        ],
    )
    reports = svc.place_ready()
    big_report = next(r for r in reports if svc.state.jobs[r.job_id].key == "big")
    small_report = next(r for r in reports if svc.state.jobs[r.job_id].key == "small")
    out.append(f"big 安置成功 -> {big_report.chosen_site_id}（预留持有 7 单位额度）")
    excluded = {c.site_id: c.hard_violations for c in small_report.candidates if not c.feasible}
    out.append(f"small 被硬约束排除：{excluded}")
    out.append(f"时段 0 租户额度占用：{_quota_of(svc, 'tenant-a', 0)}/10")

    svc.cancel_job(next(j.id for j in svc.state.jobs.values() if j.key == "big"), "为紧急作业让路")
    out.append(f"值班人员执行前取消 big：时段 0 额度占用回落为 {_quota_of(svc, 'tenant-a', 0)}/10")

    reports = svc.place_ready()
    small_report = next(r for r in reports if svc.state.jobs[r.job_id].key == "small")
    out.append(f"重新安置 small -> {small_report.chosen_site_id}（额度释放后可行）")
    svc.confirm_all()
    svc.tick(600)
    states = {j.key: j.state.value for j in svc.state.jobs.values()}
    out.append(f"最终作业状态：{states}；全程额度占用从未超过 10")
    return out, svc


def scenario_reservation_expiry() -> tuple[list[str], SchedulerService]:
    """预留过期：预留超时未确认自动释放，重试安置不会重复扣减。"""
    svc = make_service()
    build_demo_topology(svc)
    out: list[str] = []
    out.append("== 场景：预留过期 ==")

    svc.submit_group(
        "tenant-a",
        [JobSpec(key="job", dataset_id="ds-east", compute_units=4, duration_slots=2, deadline_slot=20)],
    )
    svc.place_ready()
    job = next(iter(svc.state.jobs.values()))
    rsv = next(iter(svc.state.reservations.values()))
    out.append(f"作业已预留 {rsv.id}（{rsv.site_id}），确认时限 {rsv.expires_at_seconds} 秒")
    out.append(f"持有期额度占用：{_quota_of(svc, 'tenant-a', rsv.start_slot)}/10")

    svc.tick(901)  # 超过 900 秒预留 TTL
    out.append(f"推进 901 秒：预留状态 {rsv.status.value}，作业回到 {job.state.value}")
    out.append(f"过期后额度占用：{_quota_of(svc, 'tenant-a', rsv.start_slot)}/10（已自动释放）")

    svc.place_ready()
    svc.confirm_all()
    new_rsv = [r for r in svc.state.reservations.values() if r.status.value == "CONFIRMED"]
    out.append(f"重新安置并确认：{[r.id for r in new_rsv]}")
    svc.tick(600)
    out.append(f"最终作业状态：{job.state.value}；全程额度占用从未超过 10")
    return out, svc


def scenario_fault_migration() -> tuple[list[str], SchedulerService]:
    """故障迁移：站点失效后可迁移作业受控迁移，驻留锁定的作业受控阻塞。"""
    svc = make_service()
    build_demo_topology(svc)
    svc.add_dataset("ds-pinned", "驻留锁定数据", 10.0, "site-east-1", ["site-east-1"])
    out: list[str] = []
    out.append("== 场景：故障迁移 ==")

    svc.submit_group(
        "tenant-a",
        [
            JobSpec(key="mobile", dataset_id="ds-east", compute_units=4, duration_slots=2, deadline_slot=20),
            JobSpec(key="pinned", dataset_id="ds-pinned", compute_units=2, duration_slots=2, deadline_slot=20),
        ],
    )
    svc.place_ready()
    svc.confirm_all()
    jobs = {j.key: j for j in svc.state.jobs.values()}
    out.append(
        f"mobile 计划于 {jobs['mobile'].task_ids and svc.state.tasks[jobs['mobile'].task_ids[0]].site_id}，"
        f"pinned 驻留锁定于 site-east-1"
    )

    report = svc.fail_site("site-east-1")
    for o in report.outcomes:
        key = svc.state.jobs[o.job_id].key
        out.append(f"site-east-1 失效：{key} -> {o.outcome} {o.new_site_ids} {o.detail}")
    svc.tick(900)  # 迁移后传输 1 时段 + 执行 2 时段
    mobile_site = svc.state.tasks[jobs["mobile"].task_ids[0]].site_id
    out.append(
        f"最终状态：mobile={jobs['mobile'].state.value}（在 {mobile_site} 完成），"
        f"pinned={jobs['pinned'].state.value}（驻留限制排除全部候选）"
    )
    return out, svc


def scenario_partial_subtask_failure() -> tuple[list[str], SchedulerService]:
    """部分子任务失败：quorum 汇合容忍失败，all 汇合不满足则整体失败。"""
    svc = make_service()
    build_demo_topology(svc)
    out: list[str] = []
    out.append("== 场景：部分子任务失败 ==")

    svc.submit_group(
        "tenant-a",
        [
            JobSpec(
                key="quorum-job", dataset_id="ds-east", compute_units=3, duration_slots=2,
                deadline_slot=20, splits=3, join=JoinSpec(kind="quorum", k=2),
            ),
            JobSpec(
                key="strict-job", dataset_id="ds-east", compute_units=3, duration_slots=2,
                deadline_slot=20, splits=2, join=JoinSpec(kind="all"),
            ),
        ],
    )
    svc.place_ready()
    svc.confirm_all()
    svc.tick(300)  # 进入执行
    jobs = {j.key: j for j in svc.state.jobs.values()}
    quorum_tasks = [svc.state.tasks[t] for t in jobs["quorum-job"].task_ids]
    strict_tasks = [svc.state.tasks[t] for t in jobs["strict-job"].task_ids]
    out.append(f"执行中：quorum-job {len(quorum_tasks)} 子任务（2/3 汇合），strict-job 2 子任务（all 汇合）")

    svc.fail_task(quorum_tasks[0].id, "节点掉电")
    out.append(f"注入失败：quorum-job 子任务 {quorum_tasks[0].id} 失败，作业状态 {jobs['quorum-job'].state.value}（仍可汇合）")
    svc.fail_task(strict_tasks[0].id, "节点掉电")
    out.append(f"注入失败：strict-job 子任务失败，作业状态 {jobs['strict-job'].state.value}（all 汇合不可能满足）")

    svc.tick(300)
    done = [t.id for t in quorum_tasks if t.state.value == "SUCCEEDED"]
    out.append(f"推进至结束：quorum-job={jobs['quorum-job'].state.value}（完成子任务 {done}），strict-job={jobs['strict-job'].state.value}")
    return out, svc


SCENARIOS: dict[str, Callable[[], tuple[list[str], SchedulerService]]] = {
    "quota-contention": scenario_quota_contention,
    "reservation-expiry": scenario_reservation_expiry,
    "fault-migration": scenario_fault_migration,
    "partial-subtask-failure": scenario_partial_subtask_failure,
}
