"""用例服务：作业组提交、安置、预留确认、取消、虚拟时间推进、
站点失效后的受控迁移以及决策解释。

所有资源占用由预留记录推导（见 domain.constraints），本服务只负责
状态机迁移与预留记录的创建/状态变更，从而保证：
- 任何重试都不会重复扣减额度（确认、迁移、故障重放均幂等）；
- 拆分作业的子任务只有满足汇合条件才可进入完成态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain.engine import evaluate_task
from ..domain.errors import DomainError
from ..domain.models import (
    JOB_CANCELLABLE_STATES,
    JOB_TERMINAL_STATES,
    TASK_ACTIVE_STATES,
    Dataset,
    DecisionReport,
    Job,
    JobGroup,
    JobSpec,
    JobState,
    JoinSpec,
    Link,
    Park,
    Reservation,
    ReservationStatus,
    Site,
    SiteStatus,
    State,
    Task,
    TaskState,
    Tenant,
)
from .ports import Clock, EventSink, IdGenerator


@dataclass
class MigrationOutcome:
    job_id: str
    outcome: str  # MIGRATED_AUTO_CONFIRMED | MIGRATED_AWAITING_CONFIRM | BLOCKED
    new_site_ids: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class MigrationReport:
    site_id: str
    outcomes: list[MigrationOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "outcomes": [vars(o) for o in self.outcomes],
        }


class SchedulerService:
    def __init__(self, state: State, clock: Clock, ids: IdGenerator, events: EventSink) -> None:
        self.state = state
        self.clock = clock
        self.ids = ids
        self.events = events

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        self.state.now_seconds = self.clock.now()

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        self.events.emit(kind, message, data)

    def _job(self, job_id: str) -> Job:
        job = self.state.jobs.get(job_id)
        if job is None:
            raise DomainError("JOB_NOT_FOUND", f"作业不存在: {job_id}")
        return job

    def _task(self, task_id: str) -> Task:
        task = self.state.tasks.get(task_id)
        if task is None:
            raise DomainError("TASK_NOT_FOUND", f"子任务不存在: {task_id}")
        return task

    def _job_tasks(self, job: Job) -> list[Task]:
        return [self.state.tasks[tid] for tid in job.task_ids]

    def _derive_job_state(self, job: Job) -> JobState:
        """由子任务状态推导进行中的作业状态（终态作业不调用）。"""
        sts = {t.state for t in self._job_tasks(job)}
        if sts & {TaskState.RUNNING, TaskState.JOIN_WAIT}:
            return JobState.RUNNING
        if sts & {TaskState.SCHEDULED}:
            return JobState.SCHEDULED
        if sts == {TaskState.RESERVED}:
            return JobState.RESERVED
        return JobState.READY

    # ------------------------------------------------------------------
    # 拓扑与租户管理
    # ------------------------------------------------------------------

    def add_park(self, park_id: str, name: str, energy_cap_per_slot: float) -> Park:
        if park_id in self.state.parks:
            raise DomainError("PARK_EXISTS", f"园区已存在: {park_id}")
        park = Park(id=park_id, name=name, energy_cap_per_slot=energy_cap_per_slot)
        self.state.parks[park_id] = park
        return park

    def add_site(
        self,
        site_id: str,
        name: str,
        region: str,
        park_id: str,
        energy_tier: int,
        compute_units_per_slot: int,
        kwh_per_unit: float,
    ) -> Site:
        if site_id in self.state.sites:
            raise DomainError("SITE_EXISTS", f"站点已存在: {site_id}")
        if park_id not in self.state.parks:
            raise DomainError("PARK_NOT_FOUND", f"园区不存在: {park_id}")
        site = Site(
            id=site_id,
            name=name,
            region=region,
            park_id=park_id,
            energy_tier=energy_tier,
            compute_units_per_slot=compute_units_per_slot,
            kwh_per_unit=kwh_per_unit,
        )
        self.state.sites[site_id] = site
        return site

    def add_link(
        self, link_id: str, site_a: str, site_b: str, bandwidth_gb_per_s: float, max_concurrent: int
    ) -> Link:
        if link_id in self.state.links:
            raise DomainError("LINK_EXISTS", f"链路已存在: {link_id}")
        for s in (site_a, site_b):
            if s not in self.state.sites:
                raise DomainError("SITE_NOT_FOUND", f"站点不存在: {s}")
        link = Link(
            id=link_id,
            site_a=site_a,
            site_b=site_b,
            bandwidth_gb_per_s=bandwidth_gb_per_s,
            max_concurrent_transfers=max_concurrent,
        )
        self.state.links[link_id] = link
        return link

    def add_tenant(
        self, tenant_id: str, name: str, max_concurrent_units: int, transfer_budget_gb: float
    ) -> Tenant:
        if tenant_id in self.state.tenants:
            raise DomainError("TENANT_EXISTS", f"租户已存在: {tenant_id}")
        tenant = Tenant(
            id=tenant_id,
            name=name,
            max_concurrent_units=max_concurrent_units,
            transfer_budget_gb=transfer_budget_gb,
        )
        self.state.tenants[tenant_id] = tenant
        return tenant

    def add_dataset(
        self, dataset_id: str, name: str, size_gb: float, location_site_id: str, allowed_site_ids: list[str]
    ) -> Dataset:
        if dataset_id in self.state.datasets:
            raise DomainError("DATASET_EXISTS", f"数据集已存在: {dataset_id}")
        if location_site_id not in self.state.sites:
            raise DomainError("SITE_NOT_FOUND", f"站点不存在: {location_site_id}")
        unknown = [s for s in allowed_site_ids if s not in self.state.sites]
        if unknown:
            raise DomainError("SITE_NOT_FOUND", f"驻留站点不存在: {unknown}")
        ds = Dataset(
            id=dataset_id,
            name=name,
            size_gb=size_gb,
            location_site_id=location_site_id,
            allowed_site_ids=list(allowed_site_ids),
        )
        self.state.datasets[dataset_id] = ds
        return ds

    # ------------------------------------------------------------------
    # 作业组提交（带依赖，一次提交）
    # ------------------------------------------------------------------

    def submit_group(self, tenant_id: str, specs: list[JobSpec]) -> JobGroup:
        self._sync()
        if tenant_id not in self.state.tenants:
            raise DomainError("TENANT_NOT_FOUND", f"租户不存在: {tenant_id}")
        if not specs:
            raise DomainError("EMPTY_GROUP", "作业组不能为空")
        keys = [s.key for s in specs]
        if len(set(keys)) != len(keys):
            raise DomainError("DUPLICATE_JOB_KEY", "作业组内 key 重复")
        key_set = set(keys)
        for s in specs:
            if s.dataset_id not in self.state.datasets:
                raise DomainError("DATASET_NOT_FOUND", f"数据集不存在: {s.dataset_id}")
            unknown_deps = [d for d in s.depends_on if d not in key_set]
            if unknown_deps:
                raise DomainError("DEP_NOT_FOUND", f"作业 {s.key} 依赖未知作业: {unknown_deps}")
            if s.key in s.depends_on:
                raise DomainError("DEP_CYCLE", f"作业 {s.key} 不能依赖自身")
            if s.compute_units <= 0 or s.duration_slots <= 0:
                raise DomainError("INVALID_SPEC", f"作业 {s.key} 的算力或时长必须为正")
            if s.splits <= 0:
                raise DomainError("INVALID_SPEC", f"作业 {s.key} 的拆分数量必须为正")
            if s.join.kind not in ("all", "quorum"):
                raise DomainError("INVALID_SPEC", f"作业 {s.key} 的汇合方式须为 all 或 quorum")
            if s.join.kind == "quorum" and not (1 <= s.join.k <= s.splits):
                raise DomainError("INVALID_SPEC", f"作业 {s.key} 的汇合 quorum 超出范围")
            if s.deadline_slot < s.earliest_start_slot + s.duration_slots:
                raise DomainError("INVALID_SPEC", f"作业 {s.key} 的时限无法容纳执行时长")
        self._assert_acyclic(specs)

        group_id = self.ids.next("grp")
        job_ids: list[str] = []
        key_to_job_id: dict[str, str] = {}
        for s in specs:
            job_id = self.ids.next("job")
            key_to_job_id[s.key] = job_id
            job_ids.append(job_id)
        for s in specs:
            job_id = key_to_job_id[s.key]
            dep_ids = [key_to_job_id[d] for d in s.depends_on]
            job = Job(
                id=job_id,
                group_id=group_id,
                tenant_id=tenant_id,
                key=s.key,
                dataset_id=s.dataset_id,
                compute_units=s.compute_units,
                duration_slots=s.duration_slots,
                deadline_slot=s.deadline_slot,
                earliest_start_slot=s.earliest_start_slot,
                depends_on=dep_ids,
                join=JoinSpec(kind=s.join.kind, k=s.join.k),
                state=JobState.READY if not dep_ids else JobState.PENDING,
            )
            # 拆分：算力均分，余数分给首个子任务；数据分片均分
            base = s.compute_units // s.splits
            remainder = s.compute_units - base * s.splits
            dataset = self.state.datasets[s.dataset_id]
            shard_gb = dataset.size_gb / s.splits
            for i in range(s.splits):
                units = base + (remainder if i == 0 else 0)
                task = Task(
                    id=self.ids.next("task"),
                    job_id=job_id,
                    seq=i,
                    compute_units=units,
                    duration_slots=s.duration_slots,
                    shard_gb=shard_gb,
                )
                self.state.tasks[task.id] = task
                job.task_ids.append(task.id)
            self.state.jobs[job_id] = job
        group = JobGroup(
            id=group_id,
            tenant_id=tenant_id,
            submitted_at_slot=self.state.now_slot,
            job_ids=job_ids,
        )
        self.state.groups[group_id] = group
        self._emit(
            "GROUP_SUBMITTED",
            f"作业组 {group_id} 已提交（{len(specs)} 个作业）",
            group_id=group_id,
            tenant_id=tenant_id,
            jobs={s.key: key_to_job_id[s.key] for s in specs},
        )
        return group

    @staticmethod
    def _assert_acyclic(specs: list[JobSpec]) -> None:
        graph = {s.key: list(s.depends_on) for s in specs}
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(node: str, trail: list[str]) -> None:
            if node in done:
                return
            if node in visiting:
                raise DomainError("DEP_CYCLE", f"依赖存在环: {' -> '.join(trail + [node])}")
            visiting.add(node)
            for dep in graph[node]:
                visit(dep, trail + [node])
            visiting.discard(node)
            done.add(node)

        for key in graph:
            visit(key, [])

    # ------------------------------------------------------------------
    # 安置决策
    # ------------------------------------------------------------------

    def place_ready(self) -> list[DecisionReport]:
        """对全部 READY 作业执行安置，返回所有决策报告。"""
        self._sync()
        reports: list[DecisionReport] = []
        ready = sorted(
            (j for j in self.state.jobs.values() if j.state == JobState.READY),
            key=lambda j: j.id,
        )
        for job in ready:
            reports.extend(self.place_job(job.id))
        return reports

    def place_job(self, job_id: str) -> list[DecisionReport]:
        """为一个 READY 作业的全部待安置子任务创建预留。

        拆分作业的整体安置是原子的：任一子任务无可行站点时，
        本次为该作业已创建的预留全部回滚，作业保持 READY 可重试。
        """
        self._sync()
        job = self._job(job_id)
        if job.state == JobState.RESERVED:
            return []  # 幂等：已持有预留，重复安置不产生新扣减
        if job.state != JobState.READY:
            raise DomainError("JOB_NOT_READY", f"作业 {job_id} 当前状态 {job.state.value} 不可安置")

        reports: list[DecisionReport] = []
        created: list[str] = []
        pending = [t for t in self._job_tasks(job) if t.state == TaskState.PENDING]
        for task in sorted(pending, key=lambda t: t.seq):
            report, plan = evaluate_task(self.state, job, task, self.ids.next("dec"))
            if plan is None:
                # 回滚本次为该作业创建的预留，保证不产生残留扣减
                for rid in created:
                    self.state.reservations[rid].status = ReservationStatus.CANCELLED
                for t in self._job_tasks(job):
                    if t.reservation_id in created:
                        t.state = TaskState.PENDING
                        t.site_id = None
                        t.reservation_id = None
                reports.append(report)
                self.state.decisions.extend(reports)
                self._emit(
                    "PLACEMENT_FAILED",
                    f"作业 {job.id} 安置失败：子任务 {task.id} 无可行站点",
                    job_id=job.id,
                    task_id=task.id,
                    decision_id=report.id,
                )
                return reports
            reservation = self._create_reservation(job, task, plan)
            report.reservation_id = reservation.id
            created.append(reservation.id)
            reports.append(report)

        self.state.decisions.extend(reports)
        if all(t.state == TaskState.RESERVED for t in self._job_tasks(job)):
            job.state = JobState.RESERVED
            self._emit(
                "JOB_RESERVED",
                f"作业 {job.id} 全部子任务已预留，等待确认",
                job_id=job.id,
                reservations=created,
            )
        elif job.state not in JOB_TERMINAL_STATES:
            # 混合状态（如迁移中部分子任务仍在执行）：按子任务推导
            job.state = self._derive_job_state(job)
        return reports

    def _create_reservation(self, job: Job, task: Task, plan: Any) -> Reservation:
        now = self.clock.now()
        reservation = Reservation(
            id=self.ids.next("rsv"),
            task_id=task.id,
            job_id=job.id,
            tenant_id=job.tenant_id,
            site_id=plan.site_id,
            start_slot=plan.start_slot,
            duration_slots=task.duration_slots,
            compute_units=task.compute_units,
            energy_per_slot=plan.energy_per_slot,
            transfer=plan.transfer,
            created_at_seconds=now,
            expires_at_seconds=now + self.state.settings.reservation_ttl_seconds,
        )
        self.state.reservations[reservation.id] = reservation
        task.state = TaskState.RESERVED
        task.site_id = plan.site_id
        task.reservation_id = reservation.id
        return reservation

    # ------------------------------------------------------------------
    # 预留确认与取消
    # ------------------------------------------------------------------

    def confirm_job(self, job_id: str) -> None:
        """确认作业全部持有中的预留。重复确认是幂等空操作。"""
        self._sync()
        job = self._job(job_id)
        if job.state == JobState.SCHEDULED:
            return  # 幂等
        if job.state != JobState.RESERVED:
            raise DomainError("JOB_NOT_RESERVED", f"作业 {job_id} 状态 {job.state.value} 不可确认")
        for task in self._job_tasks(job):
            if task.state == TaskState.RESERVED:
                self._confirm_reservation(task.reservation_id)  # type: ignore[arg-type]
            elif task.state in (
                TaskState.SCHEDULED,
                TaskState.RUNNING,
                TaskState.JOIN_WAIT,
                TaskState.SUCCEEDED,
            ):
                continue  # 此前已确认（例如部分过期后重确认、迁移中的混合状态）
            else:
                raise DomainError("TASK_NOT_RESERVED", f"子任务 {task.id} 未持有预留")
        job.state = JobState.SCHEDULED
        self._emit("JOB_CONFIRMED", f"作业 {job.id} 已确认，等待执行", job_id=job.id)

    def confirm_all(self) -> dict[str, Any]:
        """确认所有处于 RESERVED 状态的作业。

        单个作业确认失败（如预留已过期）不影响其他作业；
        返回 {"confirmed": [...], "failed": [{job_id, code, message}]}。
        """
        self._sync()
        confirmed: list[str] = []
        failed: list[dict[str, str]] = []
        for job in sorted(self.state.jobs.values(), key=lambda j: j.id):
            if job.state != JobState.RESERVED:
                continue
            try:
                self.confirm_job(job.id)
                confirmed.append(job.id)
            except DomainError as e:
                failed.append({"job_id": job.id, "code": e.code, "message": e.message})
        return {"confirmed": confirmed, "failed": failed}

    def _confirm_reservation(self, reservation_id: str) -> None:
        r = self.state.reservations[reservation_id]
        if r.status == ReservationStatus.CONFIRMED:
            return  # 幂等：重复确认不会重复扣减
        if r.status != ReservationStatus.HELD:
            raise DomainError(
                "RESERVATION_NOT_HELD", f"预留 {reservation_id} 状态 {r.status.value} 不可确认"
            )
        if r.expires_at_seconds <= self.state.now_seconds:
            self._expire_reservation(r)
            raise DomainError("RESERVATION_EXPIRED", f"预留 {reservation_id} 已超时释放")
        r.status = ReservationStatus.CONFIRMED
        if r.transfer is not None:
            r.transfer_charged = True  # 传输预算在此刻转为实际扣减（仅一次）
        task = self.state.tasks[r.task_id]
        task.state = TaskState.SCHEDULED
        self._emit(
            "RESERVATION_CONFIRMED",
            f"预留 {r.id} 已确认（站点 {r.site_id}，时段 {r.start_slot}-{r.end_slot}）",
            reservation_id=r.id,
            job_id=r.job_id,
            task_id=r.task_id,
            site_id=r.site_id,
        )

    def cancel_job(self, job_id: str, reason: str = "") -> None:
        """执行前取消：仅允许尚未开始执行的作业。"""
        self._sync()
        job = self._job(job_id)
        if job.state not in JOB_CANCELLABLE_STATES:
            raise DomainError(
                "JOB_NOT_CANCELLABLE",
                f"作业 {job_id} 状态 {job.state.value}，仅执行前可取消",
            )
        for task in self._job_tasks(job):
            self._release_task_reservation(task, ReservationStatus.CANCELLED)
            if task.state not in (TaskState.FAILED,):
                task.state = TaskState.CANCELLED
        job.state = JobState.CANCELLED
        job.block_reason = reason or None
        self._emit("JOB_CANCELLED", f"作业 {job.id} 已在执行前取消", job_id=job.id, reason=reason)

    # ------------------------------------------------------------------
    # 虚拟时间推进
    # ------------------------------------------------------------------

    def tick(self, seconds: int) -> None:
        """推进虚拟时间并处理到期事件：预留超时释放、任务启停、汇合判定。"""
        if seconds < 0:
            raise DomainError("INVALID_TICK", "推进时长不能为负")
        self.clock.advance(seconds)
        self._sync()
        self._expire_due_reservations()
        self._start_due_tasks()
        self._finish_due_tasks()
        self._refresh_dependencies()
        self._emit("TICK", f"虚拟时间推进 {seconds} 秒至 {self.state.now_seconds}", seconds=seconds)

    def _expire_due_reservations(self) -> None:
        for r in sorted(self.state.reservations.values(), key=lambda x: x.id):
            if r.status == ReservationStatus.HELD and r.expires_at_seconds <= self.state.now_seconds:
                self._expire_reservation(r)

    def _expire_reservation(self, r: Reservation) -> None:
        r.status = ReservationStatus.EXPIRED
        task = self.state.tasks[r.task_id]
        if task.reservation_id == r.id:
            task.state = TaskState.PENDING
            task.site_id = None
            task.reservation_id = None
        job = self.state.jobs[r.job_id]
        if job.state not in JOB_TERMINAL_STATES and job.state != JobState.PENDING:
            job.state = self._derive_job_state(job)
        self._emit(
            "RESERVATION_EXPIRED",
            f"预留 {r.id} 超时未确认，已自动释放",
            reservation_id=r.id,
            job_id=r.job_id,
            task_id=r.task_id,
        )

    def _start_due_tasks(self) -> None:
        now_slot = self.state.now_slot
        for task in sorted(self.state.tasks.values(), key=lambda t: t.id):
            if task.state != TaskState.SCHEDULED or task.reservation_id is None:
                continue
            r = self.state.reservations[task.reservation_id]
            if r.status == ReservationStatus.CONFIRMED and r.start_slot <= now_slot:
                task.state = TaskState.RUNNING
                job = self.state.jobs[task.job_id]
                if job.state == JobState.SCHEDULED:
                    job.state = JobState.RUNNING
                self._emit(
                    "TASK_STARTED",
                    f"子任务 {task.id} 在站点 {r.site_id} 开始执行",
                    task_id=task.id,
                    job_id=task.job_id,
                    site_id=r.site_id,
                )

    def _finish_due_tasks(self) -> None:
        now_slot = self.state.now_slot
        due_job_ids: list[str] = []
        for task in sorted(self.state.tasks.values(), key=lambda t: t.id):
            if task.state != TaskState.RUNNING or task.reservation_id is None:
                continue
            r = self.state.reservations[task.reservation_id]
            if r.end_slot <= now_slot:
                # 执行完毕，先在汇合屏障等待，不直接进入完成态
                task.state = TaskState.JOIN_WAIT
                self._emit(
                    "TASK_JOIN_WAIT",
                    f"子任务 {task.id} 执行完毕，等待汇合",
                    task_id=task.id,
                    job_id=task.job_id,
                )
                if task.job_id not in due_job_ids:
                    due_job_ids.append(task.job_id)
        # 同一时刻到达屏障的子任务全部登记后再判定汇合
        for job_id in due_job_ids:
            self._evaluate_join(self.state.jobs[job_id])

    def _evaluate_join(self, job: Job) -> None:
        """汇合判定：只有满足汇合条件，子任务与作业才可进入完成态。"""
        if job.state in JOB_TERMINAL_STATES:
            return
        tasks = self._job_tasks(job)
        n = len(tasks)
        required = job.join.required(n)
        waiting = sum(1 for t in tasks if t.state in (TaskState.JOIN_WAIT, TaskState.SUCCEEDED))
        failed = sum(1 for t in tasks if t.state in (TaskState.FAILED, TaskState.CANCELLED))
        if waiting >= required:
            for t in tasks:
                if t.state == TaskState.JOIN_WAIT:
                    t.state = TaskState.SUCCEEDED
                    self._complete_task_reservation(t)
                elif t.state in TASK_ACTIVE_STATES or t.state == TaskState.PENDING:
                    # quorum 场景下多余的子任务取消并释放资源
                    t.state = TaskState.CANCELLED
                    self._release_task_reservation(t, ReservationStatus.CANCELLED)
            job.state = JobState.SUCCEEDED
            self._emit(
                "JOIN_SATISFIED",
                f"作业 {job.id} 汇合条件满足（{waiting}/{n}），进入完成态",
                job_id=job.id,
                required=required,
            )
        elif failed > n - required:
            for t in tasks:
                if t.state == TaskState.JOIN_WAIT:
                    t.state = TaskState.CANCELLED
                    self._complete_task_reservation(t)
                elif t.state in TASK_ACTIVE_STATES or t.state == TaskState.PENDING:
                    t.state = TaskState.CANCELLED
                    self._release_task_reservation(t, ReservationStatus.CANCELLED)
            job.state = JobState.FAILED
            job.block_reason = "汇合条件不可能满足"
            self._emit(
                "JOB_FAILED",
                f"作业 {job.id} 失败子任务过多，汇合条件不可能满足",
                job_id=job.id,
                failed=failed,
                required=required,
            )

    def _complete_task_reservation(self, task: Task) -> None:
        if task.reservation_id is None:
            return
        r = self.state.reservations[task.reservation_id]
        if r.status in (ReservationStatus.HELD, ReservationStatus.CONFIRMED):
            r.status = ReservationStatus.COMPLETED

    def _release_task_reservation(self, task: Task, status: ReservationStatus) -> None:
        if task.reservation_id is None:
            return
        r = self.state.reservations[task.reservation_id]
        if r.status in (ReservationStatus.HELD, ReservationStatus.CONFIRMED):
            r.status = status

    def _refresh_dependencies(self) -> None:
        for job in sorted(self.state.jobs.values(), key=lambda j: j.id):
            if job.state != JobState.PENDING:
                continue
            deps = [self.state.jobs[d] for d in job.depends_on]
            if any(d.state in (JobState.FAILED, JobState.CANCELLED, JobState.BLOCKED) for d in deps):
                job.state = JobState.BLOCKED
                job.block_reason = "依赖作业未成功完成"
                self._emit("JOB_BLOCKED", f"作业 {job.id} 因依赖失败被阻塞", job_id=job.id)
            elif all(d.state == JobState.SUCCEEDED for d in deps):
                job.state = JobState.READY
                self._emit("JOB_READY", f"作业 {job.id} 依赖就绪，可参与安置", job_id=job.id)

    # ------------------------------------------------------------------
    # 故障与受控迁移
    # ------------------------------------------------------------------

    def fail_task(self, task_id: str, reason: str = "注入故障") -> None:
        """注入子任务执行失败，用于复现部分子任务失败场景。"""
        self._sync()
        task = self._task(task_id)
        if task.state not in (TaskState.SCHEDULED, TaskState.RUNNING):
            raise DomainError(
                "TASK_NOT_ACTIVE", f"子任务 {task_id} 状态 {task.state.value}，无法注入失败"
            )
        task.state = TaskState.FAILED
        self._release_task_reservation(task, ReservationStatus.CANCELLED)
        self._emit(
            "TASK_FAILED",
            f"子任务 {task.id} 执行失败：{reason}",
            task_id=task.id,
            job_id=task.job_id,
            reason=reason,
        )
        self._evaluate_join(self.state.jobs[task.job_id])

    def fail_site(self, site_id: str) -> MigrationReport:
        """站点失效：受影响作业执行受控迁移。

        迁移是幂等的：已迁移的任务不再关联失效站点，重复调用不会产生
        新的预留或扣减。原已确认预留的传输扣减保留（传输已实际发生），
        算力持有随旧预留被取代而释放。
        """
        self._sync()
        site = self.state.sites.get(site_id)
        if site is None:
            raise DomainError("SITE_NOT_FOUND", f"站点不存在: {site_id}")

        report = MigrationReport(site_id=site_id)
        if site.status == SiteStatus.DOWN:
            return report  # 幂等：重复失效调用为空操作
        site.status = SiteStatus.DOWN
        self._emit("SITE_DOWN", f"站点 {site_id} 失效", site_id=site_id)

        # 找出受影响任务：在失效站点上仍持有资源且尚未完成执行的任务。
        # JOIN_WAIT 任务执行已完毕，不迁移重跑，由汇合屏障处理。
        affected: list[Task] = []
        migratable = (TaskState.RESERVED, TaskState.SCHEDULED, TaskState.RUNNING)
        for task in sorted(self.state.tasks.values(), key=lambda t: t.id):
            if task.site_id != site_id or task.reservation_id is None:
                continue
            if task.state not in migratable:
                continue
            r = self.state.reservations[task.reservation_id]
            if r.status in (ReservationStatus.HELD, ReservationStatus.CONFIRMED):
                affected.append(task)

        # 按作业分组，记录失效前是否已确认（决定迁移后是否自动确认）
        job_ids: list[str] = []
        for task in affected:
            if task.job_id not in job_ids:
                job_ids.append(task.job_id)
        for job_id in job_ids:
            job = self.state.jobs[job_id]
            was_confirmed = job.state in (JobState.SCHEDULED, JobState.RUNNING)
            # 释放失效站点上的旧预留，任务回到待安置（与受影响判定同一过滤）
            for task in self._job_tasks(job):
                if task.site_id != site_id or task.reservation_id is None:
                    continue
                if task.state not in migratable:
                    continue
                r = self.state.reservations[task.reservation_id]
                if r.status not in (ReservationStatus.HELD, ReservationStatus.CONFIRMED):
                    continue
                r.status = ReservationStatus.SUPERSEDED
                task.state = TaskState.PENDING
                task.site_id = None
                task.reservation_id = None
                task.attempt += 1
            if job.state in (JobState.RESERVED, JobState.SCHEDULED, JobState.RUNNING):
                job.state = JobState.READY
            outcome = self._migrate_job(job, was_confirmed)
            report.outcomes.append(outcome)
        return report

    def _migrate_job(self, job: Job, was_confirmed: bool) -> MigrationOutcome:
        """为失效作业重新安置；无可行目标时进入 BLOCKED。"""
        reports = self.place_job(job.id)
        job = self.state.jobs[job.id]
        unplaced = [t for t in self._job_tasks(job) if t.state == TaskState.PENDING]
        if unplaced:
            # 迁移失败：释放该作业其余资源并阻塞
            for task in self._job_tasks(job):
                if task.state == TaskState.JOIN_WAIT:
                    task.state = TaskState.CANCELLED
                    self._complete_task_reservation(task)
                elif task.state in TASK_ACTIVE_STATES or task.state == TaskState.PENDING:
                    task.state = TaskState.CANCELLED
                    self._release_task_reservation(task, ReservationStatus.CANCELLED)
            job.state = JobState.BLOCKED
            job.block_reason = "站点失效且无可行迁移目标"
            reasons = []
            for rep in reports:
                for cand in rep.candidates:
                    if not cand.feasible:
                        reasons.append(f"{cand.site_id}:{'/'.join(cand.hard_violations)}")
            self._emit(
                "MIGRATION_FAILED",
                f"作业 {job.id} 无可行迁移目标，进入阻塞",
                job_id=job.id,
                reasons=reasons,
            )
            return MigrationOutcome(job_id=job.id, outcome="BLOCKED", detail="; ".join(reasons))
        new_sites = sorted({t.site_id for t in self._job_tasks(job) if t.site_id})
        if was_confirmed:
            # 失效前已确认：新预留自动确认，作业保持既定计划（无缝受控迁移）
            for task in self._job_tasks(job):
                if task.state == TaskState.RESERVED:
                    self._confirm_reservation(task.reservation_id)  # type: ignore[arg-type]
            job.state = self._derive_job_state(job)
            self._emit(
                "MIGRATION_COMPLETED",
                f"作业 {job.id} 已受控迁移至 {new_sites}（自动确认）",
                job_id=job.id,
                sites=new_sites,
            )
            return MigrationOutcome(job_id=job.id, outcome="MIGRATED_AUTO_CONFIRMED", new_site_ids=new_sites)
        job.state = self._derive_job_state(job)
        self._emit(
            "MIGRATION_COMPLETED",
            f"作业 {job.id} 已受控迁移至 {new_sites}（等待确认）",
            job_id=job.id,
            sites=new_sites,
        )
        return MigrationOutcome(job_id=job.id, outcome="MIGRATED_AWAITING_CONFIRM", new_site_ids=new_sites)

    def recover_site(self, site_id: str) -> None:
        self._sync()
        site = self.state.sites.get(site_id)
        if site is None:
            raise DomainError("SITE_NOT_FOUND", f"站点不存在: {site_id}")

        site.status = SiteStatus.UP
        self._emit("SITE_UP", f"站点 {site_id} 已恢复", site_id=site_id)

    # ------------------------------------------------------------------
    # 查询与解释
    # ------------------------------------------------------------------

    def explain_job(self, job_id: str) -> list[DecisionReport]:
        """返回作业全部子任务的决策报告：硬约束排除原因与软目标排序。"""
        job = self._job(job_id)
        task_ids = set(job.task_ids)
        return [d for d in self.state.decisions if d.task_id in task_ids]

    def explain_decision(self, decision_id: str) -> DecisionReport:
        for d in self.state.decisions:
            if d.id == decision_id:
                return d
        raise DomainError("DECISION_NOT_FOUND", f"决策记录不存在: {decision_id}")

    def group_status(self, group_id: str) -> dict[str, Any]:
        group = self.state.groups.get(group_id)
        if group is None:
            raise DomainError("GROUP_NOT_FOUND", f"作业组不存在: {group_id}")
        jobs = [self.state.jobs[jid] for jid in group.job_ids]
        states = [j.state for j in jobs]
        if all(s == JobState.SUCCEEDED for s in states):
            overall = "SUCCEEDED"
        elif any(s in (JobState.FAILED, JobState.BLOCKED) for s in states):
            overall = "FAILED"
        elif any(s == JobState.CANCELLED for s in states):
            overall = "CANCELLED"
        elif any(s in (JobState.RUNNING, JobState.SCHEDULED, JobState.RESERVED) for s in states):
            overall = "RUNNING"
        else:
            overall = "PENDING"
        return {
            "group_id": group_id,
            "tenant_id": group.tenant_id,
            "state": overall,
            "jobs": [
                {
                    "job_id": j.id,
                    "key": j.key,
                    "state": j.state.value,
                    "tasks": [
                        {
                            "task_id": self.state.tasks[t].id,
                            "seq": self.state.tasks[t].seq,
                            "state": self.state.tasks[t].state.value,
                            "site_id": self.state.tasks[t].site_id,
                        }
                        for t in j.task_ids
                    ],
                }
                for j in jobs
            ],
        }
