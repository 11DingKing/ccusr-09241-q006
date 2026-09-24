"""站点失效后的受控迁移。"""

import unittest

from compute_network_scheduler.domain.constraints import UsageView
from compute_network_scheduler.domain.models import (
    JobSpec,
    JobState,
    ReservationStatus,
    SiteStatus,
    TaskState,
)
from helpers import assert_invariants, make_service, submit


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = make_service()

    def test_confirmed_job_migrates_and_completes_elsewhere(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.confirm_job(job.id)
        old_rsv = next(iter(svc.state.reservations.values()))
        self.assertEqual(old_rsv.site_id, "s1")

        report = svc.fail_site("s1")
        self.assertEqual(svc.state.sites["s1"].status, SiteStatus.DOWN)
        self.assertEqual(report.outcomes[0].outcome, "MIGRATED_AUTO_CONFIRMED")
        self.assertEqual(old_rsv.status, ReservationStatus.SUPERSEDED)
        new_rsv = [r for r in svc.state.reservations.values() if r.status == ReservationStatus.CONFIRMED]
        self.assertEqual(len(new_rsv), 1)
        self.assertNotEqual(new_rsv[0].site_id, "s1")
        self.assertEqual(job.state, JobState.SCHEDULED)

        svc.tick(1200)  # 传输 + 执行
        self.assertEqual(job.state, JobState.SUCCEEDED)
        assert_invariants(svc.state)

    def test_held_job_migrates_to_awaiting_confirm(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        report = svc.fail_site("s1")
        self.assertEqual(report.outcomes[0].outcome, "MIGRATED_AWAITING_CONFIRM")
        self.assertEqual(job.state, JobState.RESERVED)
        svc.confirm_job(job.id)
        self.assertEqual(job.state, JobState.SCHEDULED)
        assert_invariants(svc.state)

    def test_unmigratable_job_blocked_with_reasons(self) -> None:
        svc = self.svc
        job = submit(svc, dataset_id="d2", compute_units=2, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.confirm_job(job.id)
        self.assertEqual(next(iter(svc.state.reservations.values())).site_id, "s2")
        report = svc.fail_site("s2")
        self.assertEqual(report.outcomes[0].outcome, "BLOCKED")
        self.assertIn("DATA_RESIDENCY", report.outcomes[0].detail)
        self.assertEqual(job.state, JobState.BLOCKED)
        assert_invariants(svc.state)  # 阻塞后不得持有资源

    def test_running_job_migrates_with_attempt_increment(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.tick(300)
        task = svc.state.tasks[job.task_ids[0]]
        self.assertEqual(task.state, TaskState.RUNNING)
        svc.fail_site("s1")
        self.assertEqual(task.attempt, 2)
        self.assertEqual(task.state, TaskState.SCHEDULED)
        svc.tick(1200)
        self.assertEqual(job.state, JobState.SUCCEEDED)
        assert_invariants(svc.state)

    def test_migration_respects_tenant_quota(self) -> None:
        svc = self.svc
        # t2 额度 4：迁移期间旧持有先释放，不得出现 8 单位的瞬时占用
        svc.add_dataset("d3", "小数据", 1.0, "s1", ["s1", "s2", "s3"])
        job = submit(svc, tenant="t2", dataset_id="d3", compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.fail_site("s1")
        usage = UsageView(svc.state)
        for slot, used in usage.tenant["t2"].items():
            self.assertLessEqual(used, 4)
        self.assertEqual(job.state, JobState.SCHEDULED)
        assert_invariants(svc.state)

    def test_split_job_partially_affected(self) -> None:
        svc = self.svc
        from compute_network_scheduler.domain.models import JoinSpec

        # 12 单位拆 2 份、时限仅允许立即开始：s1 容纳不下两份，子任务分布到不同站点
        svc.state.tenants["t1"].max_concurrent_units = 12
        job = submit(
            svc, compute_units=12, duration_slots=2, deadline_slot=3,
            splits=2, join=JoinSpec(kind="all"),
        )
        svc.place_ready()
        svc.confirm_job(job.id)
        sites = {svc.state.tasks[t].site_id for t in job.task_ids}
        self.assertEqual(len(sites), 2, "前置条件：子任务应分布在两个站点")
        victim, survivor_site = sorted(sites)
        svc.fail_site(victim)
        self.assertEqual(job.state, JobState.SCHEDULED)
        for t in job.task_ids:
            task = svc.state.tasks[t]
            self.assertNotEqual(task.site_id, victim)
            self.assertEqual(task.state, TaskState.SCHEDULED)
        svc.tick(1200)
        self.assertEqual(job.state, JobState.SUCCEEDED)
        assert_invariants(svc.state)

    def test_join_wait_task_not_migrated_on_site_failure(self) -> None:
        svc = self.svc
        from compute_network_scheduler.domain.models import JoinSpec

        # 两个子任务排队使用 s1：task1 时段0-2，task2 时段2-4
        svc.state.tenants["t1"].max_concurrent_units = 12
        job = submit(
            svc, compute_units=12, duration_slots=2, deadline_slot=10,
            splits=2, join=JoinSpec(kind="all"),
        )
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.tick(600)  # task1 执行完毕进入 JOIN_WAIT，task2 开始执行
        tasks = sorted((svc.state.tasks[t] for t in job.task_ids), key=lambda t: t.id)
        self.assertEqual(tasks[0].state, TaskState.JOIN_WAIT)
        self.assertEqual(tasks[1].state, TaskState.RUNNING)
        waiting_reservation = tasks[0].reservation_id

        svc.fail_site("s1")
        # 已到达汇合屏障的子任务不迁移重跑；执行中的子任务受控迁移
        self.assertEqual(tasks[0].state, TaskState.JOIN_WAIT)
        self.assertEqual(tasks[0].reservation_id, waiting_reservation)
        self.assertEqual(tasks[1].attempt, 2)
        svc.tick(900)
        self.assertEqual(job.state, JobState.SUCCEEDED)
        self.assertEqual(tasks[0].state, TaskState.SUCCEEDED)
        assert_invariants(svc.state)

    def test_recover_site_restores_capacity(self) -> None:
        svc = self.svc
        svc.fail_site("s1")
        svc.recover_site("s1")
        self.assertEqual(svc.state.sites["s1"].status, SiteStatus.UP)
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        rsv = next(iter(svc.state.reservations.values()))
        self.assertEqual(rsv.site_id, "s1")  # 恢复后可再次安置
        assert_invariants(svc.state)


if __name__ == "__main__":
    unittest.main()
