"""测试共享助手：标准拓扑构造与全局不变量断言。

assert_invariants 在任意操作序列之后都可调用，守护以下不变量：
- 任一站点的任一时段算力占用不超过容量；
- 任一园区的任一时段能耗不超过上限；
- 任一链路的任一时段并发传输不超过上限；
- 任一租户的任一时段并发额度与累计传输预算不被突破；
- 每个子任务至多一条持有中的预留（重试不重复扣减的结构保证）；
- 拆分作业的子任务只有满足汇合条件才可进入完成态；
- 终态作业不持有任何资源。
"""

from __future__ import annotations

from compute_network_scheduler.application.services import SchedulerService
from compute_network_scheduler.domain.constraints import UsageView, tenant_transfer_usage
from compute_network_scheduler.domain.models import (
    RESERVATION_HOLDING_STATES,
    JOB_TERMINAL_STATES,
    JobState,
    JoinSpec,
    State,
    TaskState,
)
from compute_network_scheduler.infrastructure.ports_impl import (
    ManualClock,
    SequentialIds,
    StateEventSink,
)


def make_service(state: State | None = None) -> SchedulerService:
    """构造带标准测试拓扑的服务。

    拓扑：park-a(能耗上限50) / park-b(上限24)
    s1: park-a 档位1 10单位 1.0kwh；s2: park-a 档位2 6单位 2.0kwh；s3: park-b 档位3 8单位 1.0kwh
    链路：l12(1GB/s,2并发) l13(2GB/s,1并发) l23(0.5GB/s,1并发)
    租户：t1(并发10,预算100GB) t2(并发4,预算20GB)
    数据：d1 30GB@s1 可驻留三站；d2 10GB@s2 仅 s2
    """
    state = state if state is not None else State()
    clock = ManualClock(state.now_seconds)
    svc = SchedulerService(state, clock, SequentialIds(state), StateEventSink(state, clock))
    svc.add_park("park-a", "园区A", 50.0)
    svc.add_park("park-b", "园区B", 24.0)
    svc.add_site("s1", "站点1", "east", "park-a", 1, 10, 1.0)
    svc.add_site("s2", "站点2", "east", "park-a", 2, 6, 2.0)
    svc.add_site("s3", "站点3", "west", "park-b", 3, 8, 1.0)
    svc.add_link("l12", "s1", "s2", 1.0, 2)
    svc.add_link("l13", "s1", "s3", 2.0, 1)
    svc.add_link("l23", "s2", "s3", 0.5, 1)
    svc.add_tenant("t1", "租户一", 10, 100.0)
    svc.add_tenant("t2", "租户二", 4, 20.0)
    svc.add_dataset("d1", "数据一", 30.0, "s1", ["s1", "s2", "s3"])
    svc.add_dataset("d2", "数据二", 10.0, "s2", ["s2"])
    return svc


def submit(svc: SchedulerService, tenant: str = "t1", **overrides):
    """提交单作业组的便捷函数。"""
    from compute_network_scheduler.domain.models import JobSpec

    spec = JobSpec(
        key=overrides.get("key", "j"),
        dataset_id=overrides.get("dataset_id", "d1"),
        compute_units=overrides.get("compute_units", 4),
        duration_slots=overrides.get("duration_slots", 2),
        deadline_slot=overrides.get("deadline_slot", 20),
        depends_on=overrides.get("depends_on", []),
        earliest_start_slot=overrides.get("earliest_start_slot", 0),
        splits=overrides.get("splits", 1),
        join=overrides.get("join", JoinSpec()),
    )
    svc.submit_group(tenant, [spec])
    return svc.state.jobs[[j.id for j in svc.state.jobs.values()][-1]]


def assert_invariants(state: State) -> None:
    """全局不变量：任何时刻都必须成立，违反即断言失败。"""
    usage = UsageView(state)

    # 资源上限：算力 / 能耗 / 带宽 / 租户并发额度
    for site_id, slots in usage.compute.items():
        cap = state.sites[site_id].compute_units_per_slot
        for slot, used in slots.items():
            assert used <= cap, f"站点 {site_id} 时段 {slot} 算力超占: {used}>{cap}"
    for park_id, slots in usage.energy.items():
        cap = state.parks[park_id].energy_cap_per_slot
        for slot, used in slots.items():
            assert used <= cap + 1e-9, f"园区 {park_id} 时段 {slot} 能耗超限: {used}>{cap}"
    for link_id, slots in usage.link.items():
        cap = state.links[link_id].max_concurrent_transfers
        for slot, used in slots.items():
            assert used <= cap, f"链路 {link_id} 时段 {slot} 并发传输超限: {used}>{cap}"
    for tenant_id, slots in usage.tenant.items():
        cap = state.tenants[tenant_id].max_concurrent_units
        for slot, used in slots.items():
            assert used <= cap, f"租户 {tenant_id} 时段 {slot} 额度超占: {used}>{cap}"

    # 传输预算：已扣减 + 暂扣不超过预算
    for tenant_id, tenant in state.tenants.items():
        consumed, held = tenant_transfer_usage(state, tenant_id)
        assert consumed + held <= tenant.transfer_budget_gb + 1e-9, (
            f"租户 {tenant_id} 传输预算超占: {consumed}+{held}>{tenant.transfer_budget_gb}"
        )

    # 每个子任务至多一条持有中的预留（不重复扣减的结构保证）
    holding_by_task: dict[str, int] = {}
    for r in state.reservations.values():
        if r.status in RESERVATION_HOLDING_STATES:
            holding_by_task[r.task_id] = holding_by_task.get(r.task_id, 0) + 1
    for task_id, n in holding_by_task.items():
        assert n == 1, f"子任务 {task_id} 持有 {n} 条预留，存在重复扣减"

    # 预留与任务互相引用一致
    for r in state.reservations.values():
        if r.status in RESERVATION_HOLDING_STATES:
            task = state.tasks[r.task_id]
            assert task.reservation_id == r.id, f"预留 {r.id} 与任务 {r.task_id} 引用不一致"

    # 汇合不变量：子任务进入完成态 ⇨ 所属作业已进入完成态且汇合条件满足
    for job in state.jobs.values():
        tasks = [state.tasks[tid] for tid in job.task_ids]
        n = len(tasks)
        required = job.join.required(n)
        succeeded = [t for t in tasks if t.state == TaskState.SUCCEEDED]
        if succeeded:
            assert job.state == JobState.SUCCEEDED, (
                f"作业 {job.id} 的子任务已完成但作业未进入完成态"
            )
            assert len(succeeded) >= required, (
                f"作业 {job.id} 完成子任务数 {len(succeeded)} 少于汇合要求 {required}"
            )
        if job.state == JobState.SUCCEEDED:
            assert len(succeeded) + sum(
                1 for t in tasks if t.state == TaskState.JOIN_WAIT
            ) >= required or len(succeeded) >= required, f"作业 {job.id} 未满足汇合条件却完成"

    # 终态作业不持有任何资源
    for job in state.jobs.values():
        if job.state in JOB_TERMINAL_STATES:
            for tid in job.task_ids:
                assert tid not in holding_by_task, f"终态作业 {job.id} 仍持有资源"
