"""额度与预算的幂等性：任何重试都不能重复扣减。"""

import unittest

from compute_network_scheduler.domain.constraints import UsageView, tenant_transfer_usage
from compute_network_scheduler.domain.models import (
    RESERVATION_HOLDING_STATES,
    JobSpec,
    ReservationStatus,
)
from helpers import assert_invariants, make_service, submit


def holding_reservations(state) -> list:
    return [r for r in state.reservations.values() if r.status in RESERVATION_HOLDING_STATES]


class NoDoubleDeductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = make_service()

    def _tenant_units(self, tenant: str, slot: int) -> int:
        return UsageView(self.svc.state).tenant[tenant][slot]

    def test_double_place_creates_single_reservation(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=10)
        svc.place_job(job.id)
        svc.place_job(job.id)  # 重试：幂等空操作
        svc.place_ready()  # 再次触发：作业已 RESERVED，跳过
        self.assertEqual(len(holding_reservations(svc.state)), 1)
        self.assertEqual(self._tenant_units("t1", 0), 4)
        assert_invariants(svc.state)

    def test_double_confirm_charges_transfer_once(self) -> None:
        svc = self.svc
        svc.state.datasets["d1"].allowed_site_ids = ["s3"]  # 强制传输 30GB
        job = submit(svc, compute_units=2, duration_slots=2, deadline_slot=10)
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.confirm_job(job.id)  # 重试：幂等
        svc.confirm_all()  # 批量重试：幂等
        consumed, held = tenant_transfer_usage(svc.state, "t1")
        self.assertEqual(consumed, 30.0)
        self.assertEqual(held, 0.0)
        assert_invariants(svc.state)

    def test_retry_after_expiry_does_not_accumulate(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.tick(901)  # 过期释放
        svc.place_ready()  # 重试安置
        svc.confirm_all()
        # 只有新预留在持有；旧预留已过期不占用
        self.assertEqual(len(holding_reservations(svc.state)), 1)
        consumed, held = tenant_transfer_usage(svc.state, "t1")
        self.assertEqual(consumed, 0.0)  # 数据本地，无传输
        assert_invariants(svc.state)

    def test_retry_after_cancel_does_not_accumulate(self) -> None:
        svc = self.svc
        svc.state.datasets["d1"].allowed_site_ids = ["s3"]
        job = submit(svc, compute_units=2, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.cancel_job(job.id)  # 未确认即取消：传输预算不应扣减
        consumed, held = tenant_transfer_usage(svc.state, "t1")
        self.assertEqual((consumed, held), (0.0, 0.0))
        assert_invariants(svc.state)

    def test_fail_site_is_idempotent(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.confirm_job(job.id)
        before = len(svc.state.reservations)
        report1 = svc.fail_site("s1")
        after_first = len(svc.state.reservations)
        report2 = svc.fail_site("s1")  # 重放：不得产生新预留或扣减
        self.assertEqual(report2.outcomes, [])
        self.assertEqual(len(svc.state.reservations), after_first)
        self.assertGreater(after_first, before)  # 第一次确实发生了迁移
        self.assertEqual(len(holding_reservations(svc.state)), 1)
        assert_invariants(svc.state)

    def test_migration_releases_old_compute_hold(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.fail_site("s1")
        # 旧预留被取代，新预留持有：任一时刻额度占用不超过单次 4 单位
        for slot in range(0, 6):
            self.assertLessEqual(self._tenant_units("t1", slot), 4)
        assert_invariants(svc.state)

    def test_migration_transfer_charged_per_actual_transfer(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=2, duration_slots=2, deadline_slot=20)  # 数据本地 s1
        svc.place_ready()
        svc.confirm_job(job.id)
        consumed_before, _ = tenant_transfer_usage(svc.state, "t1")
        self.assertEqual(consumed_before, 0.0)  # 本地无传输
        svc.fail_site("s1")  # 迁移到非本地站点，发生真实传输
        consumed_after, _ = tenant_transfer_usage(svc.state, "t1")
        self.assertEqual(consumed_after, 30.0)  # 仅新传输扣减一次
        svc.fail_site("s1")  # 重放不再扣减
        consumed_replay, _ = tenant_transfer_usage(svc.state, "t1")
        self.assertEqual(consumed_replay, 30.0)
        assert_invariants(svc.state)

    def test_task_failure_then_replace_single_hold(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=4, duration_slots=2, deadline_slot=20)
        svc.place_ready()
        svc.confirm_job(job.id)
        svc.tick(300)
        task = next(iter(svc.state.tasks.values()))
        svc.fail_task(task.id, "模拟崩溃")
        # 失败释放并发额度
        self.assertEqual(len(holding_reservations(svc.state)), 0)
        assert_invariants(svc.state)

    def test_quota_contention_sequence_never_exceeds_cap(self) -> None:
        svc = self.svc
        svc.submit_group(
            "t1",
            [
                JobSpec(key="big", dataset_id="d1", compute_units=7, duration_slots=3, deadline_slot=3),
                JobSpec(key="small", dataset_id="d1", compute_units=5, duration_slots=2, deadline_slot=3),
            ],
        )
        svc.place_ready()
        assert_invariants(svc.state)
        jobs = {j.key: j for j in svc.state.jobs.values()}
        svc.cancel_job(jobs["big"].id)
        assert_invariants(svc.state)
        svc.place_ready()
        svc.confirm_all()
        for slot in range(0, 4):
            self.assertLessEqual(self._tenant_units("t1", slot), 10)
        svc.tick(600)
        assert_invariants(svc.state)
        self.assertEqual(jobs["small"].state.value, "SUCCEEDED")


if __name__ == "__main__":
    unittest.main()
