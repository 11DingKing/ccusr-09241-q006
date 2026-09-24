"""预留生命周期：预留-确认、超时自动释放、执行前取消。"""

import unittest

from compute_network_scheduler.domain.errors import DomainError
from compute_network_scheduler.domain.models import JobState, ReservationStatus, TaskState
from helpers import assert_invariants, make_service, submit


class ReservationLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = make_service()

    def test_reserve_confirm_run_complete(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=10)
        svc.place_ready()
        self.assertEqual(job.state, JobState.RESERVED)
        rsv = next(iter(svc.state.reservations.values()))
        self.assertEqual(rsv.status, ReservationStatus.HELD)
        self.assertEqual(rsv.expires_at_seconds, 900)

        svc.confirm_job(job.id)
        self.assertEqual(job.state, JobState.SCHEDULED)
        self.assertEqual(rsv.status, ReservationStatus.CONFIRMED)

        svc.tick(300)  # 时段 1：开始执行
        task = next(iter(svc.state.tasks.values()))
        self.assertEqual(task.state, TaskState.RUNNING)
        svc.tick(300)  # 时段 2：执行完毕
        self.assertEqual(task.state, TaskState.SUCCEEDED)
        self.assertEqual(job.state, JobState.SUCCEEDED)
        self.assertEqual(rsv.status, ReservationStatus.COMPLETED)
        assert_invariants(svc.state)

    def test_reservation_expires_and_releases_everything(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        rsv = next(iter(svc.state.reservations.values()))
        svc.tick(900)  # 恰好到期
        self.assertEqual(rsv.status, ReservationStatus.EXPIRED)
        self.assertEqual(job.state, JobState.READY)
        task = next(iter(svc.state.tasks.values()))
        self.assertEqual(task.state, TaskState.PENDING)
        self.assertIsNone(task.reservation_id)
        assert_invariants(svc.state)

    def test_confirm_after_expiry_is_rejected(self) -> None:
        svc = self.svc
        job = submit(svc)
        svc.place_ready()
        svc.tick(901)
        with self.assertRaises(DomainError) as ctx:
            svc.confirm_job(job.id)
        self.assertEqual(ctx.exception.code, "JOB_NOT_RESERVED")

    def test_confirm_expired_reservation_marks_expired(self) -> None:
        svc = self.svc
        job = submit(svc)
        svc.place_ready()
        rsv = next(iter(svc.state.reservations.values()))
        svc.clock.advance(901)  # 不经过 tick，直接确认
        with self.assertRaises(DomainError) as ctx:
            svc.confirm_job(job.id)
        self.assertEqual(ctx.exception.code, "RESERVATION_EXPIRED")
        self.assertEqual(rsv.status, ReservationStatus.EXPIRED)
        assert_invariants(svc.state)

    def test_cancel_before_execution_releases_holds(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=10)
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.cancel_job(job.id, "计划变更")
        self.assertEqual(job.state, JobState.CANCELLED)
        rsv = next(iter(svc.state.reservations.values()))
        self.assertEqual(rsv.status, ReservationStatus.CANCELLED)
        assert_invariants(svc.state)

    def test_cancel_reserved_job(self) -> None:
        svc = self.svc
        job = submit(svc)
        svc.place_ready()
        svc.cancel_job(job.id)
        self.assertEqual(job.state, JobState.CANCELLED)
        assert_invariants(svc.state)

    def test_cancel_running_job_is_rejected(self) -> None:
        svc = self.svc
        job = submit(svc, duration_slots=2, deadline_slot=10)
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.tick(300)
        with self.assertRaises(DomainError) as ctx:
            svc.cancel_job(job.id)
        self.assertEqual(ctx.exception.code, "JOB_NOT_CANCELLABLE")
        assert_invariants(svc.state)

    def test_expired_reservation_frees_quota_for_others(self) -> None:
        svc = self.svc
        # t2 额度 4：j1 占满，j2 被排除；j1 过期释放后 j2 可安置
        from compute_network_scheduler.domain.models import JobSpec

        svc.state.settings.reservation_ttl_seconds = 100
        svc.add_dataset("d3", "小数据", 1.0, "s1", ["s1", "s2", "s3"])
        svc.submit_group(
            "t2",
            [
                JobSpec(key="j1", dataset_id="d3", compute_units=4, duration_slots=3, deadline_slot=4),
                JobSpec(key="j2", dataset_id="d3", compute_units=4, duration_slots=3, deadline_slot=4),
            ],
        )
        svc.place_ready()
        jobs = {j.key: j for j in svc.state.jobs.values()}
        self.assertEqual(jobs["j1"].state, JobState.RESERVED)
        self.assertEqual(jobs["j2"].state, JobState.READY)
        svc.tick(101)  # j1 的预留过期释放
        svc.place_job(jobs["j2"].id)
        self.assertEqual(jobs["j2"].state, JobState.RESERVED)
        assert_invariants(svc.state)


if __name__ == "__main__":
    unittest.main()
