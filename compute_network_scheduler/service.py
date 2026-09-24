"""调度应用服务。

承载作业组完整生命周期：

    submit（规划）→ reserve（预留/扣额度）→ confirm（确认）
        → 时间推进中执行 → 屏障汇合 → completed
    reserve 超时 → 自动释放、退还额度
    执行前 cancel → 释放并退还
    站点失效 / 子任务失败 → 受控迁移（attempt+1，旧账幂等退还，新账重扣）

所有变更结束后原子落盘；额度扣减与容量占用都带幂等键，任何重试
（同一 attempt）都不会重复扣减。
"""

from __future__ import annotations

from . import explanation as expl
from .clock import VirtualClock
from .enums import GroupState, ReservationState, SiteStatus, TaskState
from .errors import (
    NotFoundError,
    PlacementImpossibleError,
    ReservationExpiredError,
    StateConflictError,
    ValidationError,
)
from .ledgers import CapacityLedger
from .models import (
    GroupReservation,
    GroupSpec,
    JobGroup,
    Placement,
    ReservationLine,
    TaskRuntime,
)
from .planner import Planner
from .repository import InMemoryRepository, Repository, ServiceState
from .topology import Topology


def quota_key(group_id: str, task_id: str, attempt: int) -> str:
    """额度幂等键：同一作业组/子任务/尝试序号全局唯一。"""
    return f"{group_id}:{task_id}:a{attempt}"


class SchedulerService:
    def __init__(
        self,
        repository: Repository | None = None,
        state: ServiceState | None = None,
        planner_weights: dict[str, float] | None = None,
    ) -> None:
        self.repo = repository or InMemoryRepository()
        loaded = state or self.repo.load()
        self.state = loaded or ServiceState()
        self.planner = Planner(
            self.state.topology, self.state.capacity, self.state.quota,
            self.state.clock, weights=planner_weights,
        )
        self._persist()

    # ==================================================================
    # 配置入口
    # ==================================================================
    @property
    def topology(self) -> Topology:
        return self.state.topology

    @property
    def clock(self) -> VirtualClock:
        return self.state.clock

    def _next_id(self, kind: str, prefix: str) -> str:
        n = self.state.counters.get(kind, 0) + 1
        self.state.counters[kind] = n
        return f"{prefix}-{n:04d}"

    def _persist(self) -> None:
        self.repo.save(self.state)

    # ==================================================================
    # 提交与规划
    # ==================================================================
    def submit_group(self, spec: GroupSpec) -> JobGroup:
        """校验并提交作业组，立即做一次完整规划，落盘解释但不占容量/额度。"""
        self._validate_spec(spec)
        if spec.group_id in self.state.groups:
            raise ValidationError(f"作业组 {spec.group_id} 已存在")
        if spec.tenant_id not in self.topology.tenants:
            raise ValidationError(f"租户 {spec.tenant_id} 未注册")
        for t in spec.tasks:
            for ds in t.inputs:
                if ds not in self.topology.datasets:
                    raise ValidationError(f"任务 {t.task_id} 引用了不存在的数据集 {ds}")

        group = JobGroup(spec=spec, created_at=self.clock.now)
        group.tasks = {t.task_id: TaskRuntime(spec=t) for t in spec.tasks}
        group.record(self.clock.now, GroupState.SUBMITTED, "作业组已提交")
        self.state.groups[spec.group_id] = group
        self._replan(group, kind="initial", raise_if_infeasible=True)
        group.record(self.clock.now, GroupState.PLANNED, "初始安置方案已生成")
        self._persist()
        return group

    @staticmethod
    def _validate_spec(spec: GroupSpec) -> None:
        if not spec.group_id or not spec.tenant_id:
            raise ValidationError("作业组与租户标识不能为空")
        if not spec.tasks:
            raise ValidationError("作业组至少包含一个子任务")
        if spec.reservation_ttl_slots <= 0:
            raise ValidationError("预留 TTL 必须为正整数")
        ids: set[str] = set()
        for t in spec.tasks:
            if t.task_id in ids:
                raise ValidationError(f"子任务标识重复：{t.task_id}")
            ids.add(t.task_id)
            if t.gpus <= 0 or t.duration_slots <= 0:
                raise ValidationError(f"任务 {t.task_id} 的 GPU 数与时长必须为正")
            if t.deadline is not None and t.deadline < 0:
                raise ValidationError(f"任务 {t.task_id} 的时限非法")
        idset = {t.task_id for t in spec.tasks}
        for t in spec.tasks:
            for dep in t.depends_on:
                if dep not in idset:
                    raise ValidationError(f"任务 {t.task_id} 依赖了不存在的任务 {dep}")
        Planner._topo_order(spec.tasks)  # 成环检测

    def _replan(
        self,
        group: JobGroup,
        *,
        kind: str,
        raise_if_infeasible: bool,
        tasks_to_plan: set[str] | None = None,
    ):
        """执行一次规划并保存解释；不修改容量账本。"""
        decision_id = self._next_id("decision", "dec")
        failed_sites = frozenset(
            sid for sid, s in self.topology.sites.items()
            if s.status != SiteStatus.ONLINE
        )
        result = self.planner.plan_group(
            group, decision_id, kind=kind,
            failed_sites=failed_sites, tasks_to_plan=tasks_to_plan,
        )
        self.state.decisions[decision_id] = result.explanation
        self.state.decision_index.setdefault(group.group_id, []).append(decision_id)
        group.explanation_ref = decision_id
        if not result.explanation.feasible:
            if raise_if_infeasible:
                raise PlacementImpossibleError(
                    f"作业组 {group.group_id} 无可行安置方案（决策 {decision_id}）",
                    exclusions=result.explanation.to_dict(),
                )
            return None
        # 规划结论写回任务供查询；此时尚未锁定任何容量/额度
        self._apply_plan_to_tasks(group, result.placements)
        return result

    # ==================================================================
    # 预留 / 确认 / 取消
    # ==================================================================
    def reserve(self, group_id: str) -> GroupReservation:
        """重新规划并锁定容量、幂等扣减额度，建立带 TTL 的预留。"""
        group = self._require_group(group_id)
        self.sweep_expirations()
        if group.reservation is not None and group.reservation.state == ReservationState.HELD:
            return group.reservation  # 重复请求幂等返回既有预留
        if group.state not in (GroupState.PLANNED, GroupState.SUBMITTED):
            raise StateConflictError(
                f"作业组当前状态 {group.state.value}，不能再预留")

        result = self._replan(group, kind="initial", raise_if_infeasible=True)
        assert result is not None

        now = self.clock.now
        lines: dict[str, ReservationLine] = {}
        for tid, p in result.placements.items():
            rt = group.tasks[tid]
            owner = f"{group_id}:{tid}:a{rt.attempt}"
            self._hold_placement(group, p, owner)
            # 幂等扣减：同 attempt 重试不会产生第二笔
            self.state.quota.debit(group.tenant_id, quota_key(group_id, tid, rt.attempt),
                                   p.quota_cost)
            lines[tid] = ReservationLine(
                line_id=f"{owner}", task_id=tid, attempt=rt.attempt, quota_cost=p.quota_cost,
            )
            rt.state = TaskState.RESERVED
            rt.last_event = "容量与额度已预留"

        reservation = GroupReservation(
            reservation_id=self._next_id("reservation", "rsv"),
            group_id=group_id, tenant_id=group.tenant_id, lines=lines,
            created_at=now, expires_at=now + group.spec.reservation_ttl_slots,
        )
        group.reservation = reservation
        group.record(now, GroupState.RESERVED,
                     f"预留 {reservation.reservation_id}，TTL 至 {reservation.expires_at}")
        self._persist()
        return reservation

    def confirm(self, group_id: str) -> JobGroup:
        """确认预留：锁定执行，额度不退还。重复确认幂等返回。"""
        group = self._require_group(group_id)
        self.sweep_expirations()
        rsv = group.reservation
        if rsv is None:
            raise ReservationExpiredError("预留不存在或已释放，请重新预留")
        if rsv.state == ReservationState.CONFIRMED:
            return group  # 重复确认幂等
        if rsv.state != ReservationState.HELD:
            raise ReservationExpiredError("预留已释放，请重新预留")
        if self.clock.now >= rsv.expires_at:  # 双保险
            self.sweep_expirations()
            raise ReservationExpiredError("预留已超过 TTL 自动释放")
        rsv.state = ReservationState.CONFIRMED
        for tid, line in rsv.lines.items():
            group.tasks[tid].state = TaskState.CONFIRMED
            group.tasks[tid].last_event = "预留已确认，等待执行窗口"
        group.record(self.clock.now, GroupState.CONFIRMED,
                     f"预留 {rsv.reservation_id} 已确认")
        self._persist()
        return group

    def cancel_group(self, group_id: str) -> JobGroup:
        """执行前取消：释放全部容量占用并退还额度。"""
        group = self._require_group(group_id)
        if group.state in (GroupState.CANCELLED, GroupState.COMPLETED, GroupState.FAILED):
            raise StateConflictError(f"作业组处于终态 {group.state.value}，不能取消")
        if any(rt.state in (TaskState.RUNNING, TaskState.SUCCEEDED, TaskState.COMPLETED)
               for rt in group.tasks.values()):
            raise StateConflictError("已有子任务开始执行，不能按执行前取消处理")

        self._release_group_holdings(group, refund=True, note="执行前取消")
        for rt in group.tasks.values():
            rt.state = TaskState.CANCELLED
            rt.last_event = "作业组执行前取消"
        if group.reservation is not None:
            group.reservation.state = ReservationState.RELEASED
        group.record(self.clock.now, GroupState.CANCELLED, "执行前取消，额度已退还")
        self._persist()
        return group

    # ==================================================================
    # 时间推进与执行
    # ==================================================================
    def advance(self, slots: int) -> dict:
        """推进虚拟时间，逐槽处理预留过期、执行与屏障汇合。"""
        if slots <= 0:
            raise ValidationError("推进槽数必须为正")
        events: list[dict] = []
        for _ in range(slots):
            now = self.clock.advance(1)
            events.extend(self._on_slot(now))
        self._persist()
        return {"now": self.clock.now, "events": events}

    def _on_slot(self, now: int) -> list[dict]:
        events: list[dict] = []
        events.extend(self.sweep_expirations())
        for group in list(self.state.groups.values()):
            if group.state in (GroupState.CANCELLED, GroupState.COMPLETED, GroupState.FAILED):
                continue
            events.extend(self._tick_group(group, now))
        return events

    def sweep_expirations(self) -> list[dict]:
        """释放所有到期 HELD 预留，退还额度，作业组回到可规划态。"""
        now = self.clock.now
        events: list[dict] = []
        for group in self.state.groups.values():
            rsv = group.reservation
            if rsv is None or rsv.state != ReservationState.HELD:
                continue
            if now < rsv.expires_at:
                continue
            self._release_group_holdings(group, refund=True, note="预留 TTL 超时")
            rsv.state = ReservationState.RELEASED
            group.reservation = None
            for rt in group.tasks.values():
                rt.state = TaskState.WAITING
                rt.last_event = "预留超时自动释放"
            group.record(now, GroupState.PLANNED,
                         f"预留超时（TTL 至 {rsv.expires_at}），已自动释放")
            events.append({"type": "reservation_expired", "group_id": group.group_id,
                           "reservation_id": rsv.reservation_id, "at": now})
        if events:
            self._persist()
        return events

    def _tick_group(self, group: JobGroup, now: int) -> list[dict]:
        events: list[dict] = []
        # 依赖完成映射（SUCCEEDED 也视为依赖执行完毕，屏障在组级处理）
        done_states = {TaskState.SUCCEEDED, TaskState.COMPLETED}
        # 按拓扑序遍历，保证同槽内上游先完成、下游可同槽启动
        topo_ids = Planner._topo_order(group.spec.tasks)

        for tid in topo_ids:
            rt = group.tasks[tid]
            p = rt.placement
            if p is None:
                continue
            if rt.state == TaskState.CONFIRMED:
                deps_ok = all(
                    group.tasks[d].state in done_states for d in rt.spec.depends_on
                )
                if now >= p.start_slot and deps_ok:
                    rt.state = TaskState.RUNNING
                    rt.run_progress = 0
                    rt.last_event = f"槽 {now} 开始执行于 {p.site_id}"
                    events.append({"type": "task_started", "group_id": group.group_id,
                                   "task_id": rt.spec.task_id, "site_id": p.site_id, "at": now})
                    if group.state != GroupState.RUNNING:
                        group.record(now, GroupState.RUNNING, "首个子任务开始执行")
            if rt.state == TaskState.RUNNING:
                rt.run_progress += 1
                if rt.run_progress >= rt.spec.duration_slots:
                    rt.state = TaskState.SUCCEEDED
                    rt.last_event = f"槽 {now} 执行成功，等待屏障汇合"
                    events.append({"type": "task_succeeded", "group_id": group.group_id,
                                   "task_id": rt.spec.task_id, "at": now})

        # 屏障：全部成功才放行完成态；有不可恢复失败则整组失败，成功者停在 SUCCEEDED
        states = {rt.state for rt in group.tasks.values()}
        terminal_ok = {TaskState.SUCCEEDED, TaskState.COMPLETED}
        if states and states <= terminal_ok:
            order = Planner._topo_order(group.spec.tasks)
            completed: set[str] = set()
            # 多波传播：任务仅当其依赖全部 COMPLETED 时才进入 COMPLETED
            changed = True
            while changed:
                changed = False
                for tid in order:
                    rt = group.tasks[tid]
                    if rt.state == TaskState.SUCCEEDED and all(
                        group.tasks[d].state == TaskState.COMPLETED for d in rt.spec.depends_on
                    ):
                        rt.state = TaskState.COMPLETED
                        rt.last_event = "屏障汇合放行，进入完成态"
                        completed.add(tid)
                        changed = True
            if all(rt.state == TaskState.COMPLETED for rt in group.tasks.values()):
                group.record(now, GroupState.COMPLETED,
                             f"全部 {len(group.tasks)} 个子任务越过屏障")
                events.append({"type": "group_completed", "group_id": group.group_id, "at": now})
        elif TaskState.ABORTED in states:
            permanent = any(rt.fail_permanent and rt.state == TaskState.ABORTED
                            for rt in group.tasks.values())
            stranded = any(rt.state == TaskState.ABORTED and not rt.fail_permanent
                           for rt in group.tasks.values())
            if permanent:
                self._finalize_failure(group, now, "存在不可恢复的子任务失败，屏障无法满足")
                events.append({"type": "group_failed", "group_id": group.group_id, "at": now})
            elif stranded and group.state != GroupState.RUNNING:
                group.record(now, GroupState.RUNNING, "子任务中止，等待受控迁移")
        return events

    def _finalize_failure(self, group: JobGroup, now: int, note: str) -> None:
        """整组失败：释放所有未结束任务的占用并退还，成功者保留 SUCCEEDED 不完成。"""
        for rt in group.tasks.values():
            if rt.state in (TaskState.WAITING, TaskState.RESERVED, TaskState.CONFIRMED,
                            TaskState.READY):
                self._release_task_attempt(group, rt, refund=True)
                rt.state = TaskState.CANCELLED
                rt.last_event = "因屏障无法满足被连带取消"
        group.record(now, GroupState.FAILED, note)

    # ==================================================================
    # 故障与受控迁移
    # ==================================================================
    def fail_site(self, site_id: str) -> dict:
        """宣告站点失效：中止其上所有未完成任务（可迁移，不做永久失败标记）。

        若受影响作业组尚有 HELD 预留（故障发生在确认之前），整笔预留连带
        释放并退还，其余任务回到 WAITING，由随后的受控迁移统一重安置；
        已确认 / 执行中的作业组只中止故障站点上的任务，其余站点继续执行。
        """
        site = self.topology.require_site(site_id)
        site.status = SiteStatus.FAILED
        affected: list[str] = []
        now = self.clock.now
        for group in self.state.groups.values():
            if group.state in (GroupState.CANCELLED, GroupState.COMPLETED, GroupState.FAILED):
                continue
            hit = False
            for rt in group.tasks.values():
                p = rt.placement
                if p is None or p.site_id != site_id:
                    continue
                if rt.state in (TaskState.RUNNING, TaskState.CONFIRMED, TaskState.RESERVED,
                                TaskState.READY):
                    self._abort_for_migration(group, rt, f"站点 {site_id} 失效")
                    affected.append(f"{group.group_id}/{rt.spec.task_id}")
                    hit = True
            if hit and group.reservation is not None \
                    and group.reservation.state == ReservationState.HELD:
                for rt2 in group.tasks.values():
                    if rt2.state == TaskState.RESERVED:
                        self._release_task_attempt(group, rt2, refund=True)
                        rt2.state = TaskState.WAITING
                        rt2.last_event = "未确认预留因站点失效整体释放"
                group.reservation.state = ReservationState.RELEASED
                group.reservation = None
                group.record(now, GroupState.PLANNED,
                             f"站点 {site_id} 失效，未确认预留整体释放，等待受控迁移")
        self._persist()
        return {"site_id": site_id, "status": site.status.value,
                "affected": affected, "at": now}

    def inject_task_failure(self, group_id: str, task_id: str) -> dict:
        """故障注入：正在执行的子任务永久失败，用于复现部分子任务失败。"""
        group = self._require_group(group_id)
        rt = group.tasks.get(task_id)
        if rt is None:
            raise NotFoundError(f"子任务 {task_id} 不存在")
        if rt.state != TaskState.RUNNING:
            raise StateConflictError(
                f"子任务状态为 {rt.state.value}，仅 RUNNING 可注入执行失败")
        rt.fail_permanent = True
        self._abort_for_migration(group, rt, "执行失败（故障注入，不可恢复）")
        self._finalize_failure(group, self.clock.now, "子任务永久失败，屏障汇合条件无法满足")
        self._persist()
        return {"group_id": group_id, "task_id": task_id, "state": rt.state.value,
                "group_state": group.state.value, "at": self.clock.now}

    def _abort_for_migration(self, group: JobGroup, rt, note: str) -> None:
        self._release_task_attempt(group, rt, refund=True)
        rt.state = TaskState.ABORTED
        rt.last_event = note

    def migrate_group(self, group_id: str) -> JobGroup:
        """受控迁移：为中止任务（及其未完成的下游）重新规划安置。

        - 旧 attempt 的容量与额度已在中止时释放/退还，迁移使用 attempt+1 的新键；
        - 失效站点作为硬约束参与规划，迁移决策单独留解释；
        - 迁移安置视同已确认（故障切换的人工确认即本操作），直接进入执行排队。
        """
        group = self._require_group(group_id)
        if group.state in (GroupState.CANCELLED, GroupState.COMPLETED, GroupState.FAILED):
            raise StateConflictError(f"作业组处于终态 {group.state.value}，不能迁移")
        self.sweep_expirations()

        aborted = [tid for tid, rt in group.tasks.items()
                   if rt.state == TaskState.ABORTED and not rt.fail_permanent]
        if not aborted:
            raise StateConflictError("没有可迁移的中止子任务")

        # 扩展重规划集：
        #   中止任务 + 因预留整体释放而回到 WAITING 的任务 + 上述任务的未完成下游
        replan_set = set(aborted)
        for tid, rt in group.tasks.items():
            if rt.state == TaskState.WAITING and rt.placement is not None:
                replan_set.add(tid)
        deps_map = {t.task_id: set(t.depends_on) for t in group.spec.tasks}
        changed = True
        while changed:
            changed = False
            for tid, deps in deps_map.items():
                if tid in replan_set:
                    continue
                rt = group.tasks[tid]
                if rt.state in (TaskState.SUCCEEDED, TaskState.COMPLETED, TaskState.CANCELLED):
                    continue
                if deps & replan_set:
                    replan_set.add(tid)
                    changed = True

        # 为重规划集合中尚未中止的任务释放旧 attempt（下游换绑 / 预留释放兜底）
        for tid in replan_set - set(aborted):
            rt = group.tasks[tid]
            if rt.state in (TaskState.WAITING, TaskState.RESERVED, TaskState.CONFIRMED,
                            TaskState.READY):
                self._release_task_attempt(group, rt, refund=True)
                rt.state = TaskState.WAITING

        # attempt 递增并记录旧站点
        old_sites: dict[str, str] = {}
        for tid in replan_set:
            rt = group.tasks[tid]
            if rt.placement is not None:
                old_sites[tid] = rt.placement.site_id
            rt.attempt += 1

        result = self._replan(
            group, kind="migration", raise_if_infeasible=False, tasks_to_plan=replan_set,
        )
        if result is None:
            # 无可行迁移目标：整组失败
            self._finalize_failure(
                group, self.clock.now, "受控迁移无可行安置，屏障无法满足")
            self._persist()
            decision = (
                self.state.decisions[group.explanation_ref].to_dict()
                if group.explanation_ref else None
            )
            raise PlacementImpossibleError(
                f"作业组 {group_id} 迁移失败：无可行目标站点",
                exclusions=decision,
            )

        for tid in replan_set:
            rt = group.tasks[tid]
            p = result.placements[tid]
            p.attempt = rt.attempt
            p.migrated_from = old_sites.get(tid)
            rt.placement = p
            owner = f"{group_id}:{tid}:a{rt.attempt}"
            self._hold_placement(group, p, owner)
            self.state.quota.debit(group.tenant_id,
                                   quota_key(group_id, tid, rt.attempt), p.quota_cost)
            rt.state = TaskState.CONFIRMED
            rt.run_progress = 0
            rt.last_event = f"迁移至 {p.site_id}（第 {rt.attempt} 次尝试）"

        if group.state not in (GroupState.RUNNING, GroupState.CONFIRMED):
            group.record(self.clock.now, GroupState.CONFIRMED, "迁移安置已确认")
        else:
            group.record(self.clock.now, GroupState.RUNNING,
                         f"受控迁移完成，重安置 {sorted(replan_set)}")
        self._persist()
        return group

    def recover_site(self, site_id: str) -> dict:
        """站点恢复在线（已迁移任务不自动回迁）。"""
        site = self.topology.require_site(site_id)
        site.status = SiteStatus.ONLINE
        self._persist()
        return {"site_id": site_id, "status": site.status.value, "at": self.clock.now}

    # ==================================================================
    # 占用与额度的内部原语
    # ==================================================================
    def _apply_plan_to_tasks(self, group: JobGroup, placements: dict[str, Placement]) -> None:
        for tid, p in placements.items():
            rt = group.tasks[tid]
            rt.placement = p
            rt.attempt = p.attempt

    def _hold_placement(self, group: JobGroup, p: Placement, owner: str) -> None:
        """按安置结论写容量账本；同 owner 重复调用天然幂等。

        同一 owner 的多条传输可能在同一 (链路, 槽) 上叠加，取已有值累加后写回。
        """
        cap = self.state.capacity
        gpus = group.tasks[p.task_id].spec.gpus
        cap.hold_gpu(p.site_id, p.start_slot, p.finish_slot, gpus, owner)
        for tx in p.transfers:
            for link_id in tx.edge_ids:
                for slot in range(tx.tx_start, tx.tx_finish):
                    existing = (
                        self.state.capacity._bw.get(link_id, {})
                        .get(slot, {})
                        .get(owner, 0.0)
                    )
                    cap.hold_bw(link_id, slot, existing + tx.rate_gbps, owner)

    def _release_group_holdings(self, group: JobGroup, *, refund: bool, note: str) -> None:
        for tid, rt in group.tasks.items():
            self._release_task_attempt(group, rt, refund=refund)
        # 兜底：按前缀清理（历史 attempt）
        self.state.capacity.release_group(group.group_id)
        if refund:
            self.state.quota.refund_group(group.tenant_id, group.group_id)

    def _release_task_attempt(self, group: JobGroup, rt, *, refund: bool) -> None:
        if rt.placement is None:
            return
        owner = f"{group.group_id}:{rt.spec.task_id}:a{rt.attempt}"
        self.state.capacity.release_owner(owner)
        if refund:
            self.state.quota.refund(
                group.tenant_id, quota_key(group.group_id, rt.spec.task_id, rt.attempt)
            )

    # ==================================================================
    # 查询
    # ==================================================================
    def _require_group(self, group_id: str) -> JobGroup:
        group = self.state.groups.get(group_id)
        if group is None:
            raise NotFoundError(f"作业组 {group_id} 不存在")
        return group

    def get_group(self, group_id: str) -> JobGroup:
        return self._require_group(group_id)

    def list_groups(self) -> list[JobGroup]:
        return list(self.state.groups.values())

    def tenant_quota_view(self, tenant_id: str) -> dict:
        tenant = self.topology.tenants.get(tenant_id)
        if tenant is None:
            raise NotFoundError(f"租户 {tenant_id} 不存在")
        used = self.state.quota.used(tenant_id)
        return {"tenant_id": tenant_id, "limit": tenant.quota_limit, "used": round(used, 6),
                "available": round(tenant.quota_limit - used, 6),
                "entries": self.state.quota.to_dict().get(tenant_id, {})}

    def get_decision(self, decision_id: str) -> expl.Explanation:
        decision = self.state.decisions.get(decision_id)
        if decision is None:
            raise NotFoundError(f"决策 {decision_id} 不存在")
        return decision

    def list_decisions(self, group_id: str) -> list[expl.Explanation]:
        self._require_group(group_id)
        return [self.state.decisions[d]
                for d in self.state.decision_index.get(group_id, [])]

    def explain_last_decision(self, group_id: str) -> dict:
        """运维视图：硬约束排除清单 + 软目标排序明细。"""
        group = self._require_group(group_id)
        if not group.explanation_ref:
            return {"group_id": group_id, "decisions": []}
        decision = self.state.decisions[group.explanation_ref]
        hard: dict[str, list[dict]] = {}
        soft: dict[str, list[dict]] = {}
        for cand in decision.candidates:
            if not cand.feasible:
                hard.setdefault(cand.task_id, []).append(
                    {"site_id": cand.site_id,
                     "reasons": [{"code": c, "detail": d} for c, d in cand.hard_rejections]})
            else:
                soft.setdefault(cand.task_id, []).append({
                    "site_id": cand.site_id,
                    "chosen": cand.chosen,
                    "start_slot": cand.start_slot,
                    "finish_slot": cand.finish_slot,
                    "soft": cand.soft.to_dict() if cand.soft else None,
                })
        return {
            "group_id": group_id,
            "decision_id": decision.decision_id,
            "kind": decision.kind,
            "feasible": decision.feasible,
            "note": decision.note,
            "soft_weights": decision.soft_weights,
            "group_rejections": [{"code": c, "detail": d}
                                 for c, d in decision.group_rejections],
            "hard_exclusions": hard,
            "soft_ranking": soft,
        }

    def capacity_view(self) -> dict:
        """站点容量与当前占用快照（按槽）。"""
        view: dict[str, dict] = {}
        for sid, site in self.topology.sites.items():
            slots = sorted({int(s) for s in self.state.capacity._gpu.get(sid, {}).keys()})
            view[sid] = {
                "region": site.region,
                "status": site.status.value,
                "gpu_capacity": site.gpu_capacity,
                "effective_gpu_capacity": CapacityLedger.effective_gpu_capacity(site),
                "power_capacity_kw": site.power_capacity_kw,
                "energy_tier": site.energy_tier.value,
                "slots": {str(s): self.state.capacity.gpu_used(sid, s) for s in slots},
            }
        return view
