"""安置决策：硬约束排除与软目标排序。"""

import unittest

from compute_network_scheduler.domain.constraints import (
    COMPUTE_CAPACITY,
    DATA_RESIDENCY,
    DEADLINE,
    ENERGY_CAP,
    LINK_BANDWIDTH,
    TENANT_QUOTA,
    TRANSFER_BUDGET,
)
from compute_network_scheduler.domain.models import JobSpec, JoinSpec, State
from compute_network_scheduler.interface.scenarios import make_service as make_empty_service
from helpers import assert_invariants, make_service, submit


class HardConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = make_service()

    def _violations(self, job_id: str) -> dict[str, list[str]]:
        report = [d for d in self.svc.state.decisions if d.job_id == job_id][-1]
        return {c.site_id: c.hard_violations for c in report.candidates if not c.feasible}

    def test_data_residency_excludes_non_resident_sites(self) -> None:
        job = submit(self.svc, dataset_id="d2")  # 仅允许 s2
        self.svc.place_ready()
        violations = self._violations(job.id)
        self.assertIn(DATA_RESIDENCY, violations["s1"])
        self.assertIn(DATA_RESIDENCY, violations["s3"])
        self.assertNotIn("s2", violations)
        assert_invariants(self.svc.state)

    def test_tenant_quota_excludes_when_insufficient(self) -> None:
        self.svc.add_dataset("d3", "小数据", 1.0, "s1", ["s1", "s2", "s3"])
        job = submit(self.svc, tenant="t2", dataset_id="d3", compute_units=5)  # t2 额度仅 4
        self.svc.place_ready()
        violations = self._violations(job.id)
        self.assertTrue(violations, "应存在被排除的候选")
        for site_id in ("s1", "s2", "s3"):
            self.assertIn(TENANT_QUOTA, violations[site_id])
        self.assertEqual(job.state.value, "READY")
        self.assertEqual(len(self.svc.state.reservations), 0, "安置失败不得残留预留")
        assert_invariants(self.svc.state)

    def test_compute_capacity_excludes_oversized_job(self) -> None:
        job = submit(self.svc, compute_units=20)
        self.svc.place_ready()
        violations = self._violations(job.id)
        for site_id in ("s1", "s2", "s3"):
            self.assertIn(COMPUTE_CAPACITY, violations[site_id])

    def test_deadline_excludes_when_window_passed(self) -> None:
        job = submit(self.svc, duration_slots=2, deadline_slot=2)
        self.svc.tick(300)  # 推进到时段 1，最早开始 1，超过最晚开始 0
        self.svc.place_ready()
        violations = self._violations(job.id)
        for site_id in ("s1", "s2", "s3"):
            self.assertIn(DEADLINE, violations[site_id])

    def test_energy_cap_excludes(self) -> None:
        svc = make_empty_service()
        svc.add_park("p", "园区", 5.0)  # 每时段仅 5 千瓦时
        svc.add_site("x", "站点X", "r", "p", 1, 10, 1.0)
        svc.add_tenant("t", "租户", 10, 0.0)
        svc.add_dataset("d", "数据", 1.0, "x", ["x"])
        submit(svc, tenant="t", dataset_id="d", compute_units=6, duration_slots=1, deadline_slot=5)
        svc.place_ready()
        report = svc.state.decisions[-1]
        violations = {c.site_id: c.hard_violations for c in report.candidates if not c.feasible}
        self.assertIn(ENERGY_CAP, violations["x"])
        assert_invariants(svc.state)

    def test_transfer_budget_excludes(self) -> None:
        # t2 预算 20GB，数据 30GB 在 s1，作业驻留允许 s3 -> 必须传输 30GB
        svc = self.svc
        svc.state.datasets["d1"].allowed_site_ids = ["s3"]
        job = submit(svc, tenant="t2", dataset_id="d1", compute_units=2)
        svc.place_ready()
        violations = self._violations(job.id)
        self.assertIn(TRANSFER_BUDGET, violations["s3"])

    def test_link_bandwidth_excludes_under_contention(self) -> None:
        # 两个作业都只能从 s1 传输到 s3（l13 仅 1 并发），同一时段窗口第二个无解
        svc = self.svc
        svc.state.datasets["d1"].allowed_site_ids = ["s3"]
        svc.submit_group(
            "t1",
            [
                JobSpec(key="a", dataset_id="d1", compute_units=2, duration_slots=2, deadline_slot=3),
                JobSpec(key="b", dataset_id="d1", compute_units=2, duration_slots=2, deadline_slot=3),
            ],
        )
        svc.place_ready()
        jobs = {j.key: j for j in svc.state.jobs.values()}
        self.assertEqual(jobs["a"].state.value, "RESERVED")
        self.assertEqual(jobs["b"].state.value, "READY")
        violations = self._violations(jobs["b"].id)
        self.assertIn(LINK_BANDWIDTH, violations["s3"])
        assert_invariants(svc.state)


class SoftRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = make_service()

    def test_local_site_wins_by_data_locality(self) -> None:
        submit(self.svc, dataset_id="d1")  # 数据在 s1
        reports = self.svc.place_ready()
        report = reports[0]
        self.assertEqual(report.chosen_site_id, "s1")
        feasible = [c for c in report.candidates if c.feasible]
        self.assertEqual(len(feasible), 3)
        # 软目标全部参与排序
        for c in feasible:
            self.assertEqual(
                set(c.soft_scores),
                {"data_locality", "energy_efficiency", "completion_time", "capacity_headroom"},
            )
            self.assertIsNotNone(c.total_score)
        # 可行候选按总分降序排列
        totals = [c.total_score for c in feasible]
        self.assertEqual(totals, sorted(totals, reverse=True))

    def test_energy_tier_reflected_in_soft_scores(self) -> None:
        submit(self.svc, dataset_id="d1", compute_units=2, duration_slots=1, deadline_slot=5)
        report = self.svc.place_ready()[0]
        by_site = {c.site_id: c for c in report.candidates}
        self.assertGreater(
            by_site["s1"].soft_scores["energy_efficiency"],
            by_site["s3"].soft_scores["energy_efficiency"],
        )
        self.assertGreater(
            by_site["s2"].soft_scores["energy_efficiency"],
            by_site["s3"].soft_scores["energy_efficiency"],
        )

    def test_explain_records_hard_and_soft_factors(self) -> None:
        svc = self.svc
        svc.submit_group(
            "t1",
            [
                JobSpec(key="ok", dataset_id="d1", compute_units=4, duration_slots=2, deadline_slot=10),
                JobSpec(key="bad", dataset_id="d2", compute_units=4, duration_slots=2, deadline_slot=10),
            ],
        )
        svc.place_ready()
        jobs = {j.key: j for j in svc.state.jobs.values()}
        ok_reports = svc.explain_job(jobs["ok"].id)
        self.assertEqual(ok_reports[0].chosen_site_id, "s1")
        bad_reports = svc.explain_job(jobs["bad"].id)
        excluded = {c.site_id: c.hard_violations for c in bad_reports[0].candidates if not c.feasible}
        self.assertIn(DATA_RESIDENCY, excluded["s1"])
        self.assertIn(DATA_RESIDENCY, excluded["s3"])
        self.assertEqual(bad_reports[0].chosen_site_id, "s2")


if __name__ == "__main__":
    unittest.main()
