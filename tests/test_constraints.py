"""硬约束排除与软目标排序的测试。"""

from __future__ import annotations

import unittest

from compute_network_scheduler import explanation as expl
from compute_network_scheduler.models import GroupSpec, TaskSpec
from compute_network_scheduler.errors import PlacementImpossibleError

from tests._helpers import build_service


class HardConstraintTests(unittest.TestCase):
    def test_residency_blocks_west_site(self) -> None:
        svc = build_service()
        svc.submit_group(GroupSpec("g1", "T1", [
            TaskSpec("t1", gpus=2, duration_slots=1, inputs=("D2",), deadline=4),
        ]))
        report = svc.explain_last_decision("g1")
        blocked_west = [c for c in report["hard_exclusions"].get("t1", [])
                        if c["site_id"] == "W1"]
        self.assertTrue(blocked_west and any(
            r["code"] == expl.DATA_RESIDENCY for r in blocked_west[0]["reasons"]))
        # 可行候选仅在 east 区域
        feasible_sites = [c["site_id"] for c in report["soft_ranking"]["t1"]]
        self.assertEqual(set(feasible_sites), {"E1", "E2"})

    def test_data_unreachable_when_all_replicas_on_failed_site(self) -> None:
        svc = build_service()
        from compute_network_scheduler.models import Dataset
        svc.topology.add_dataset(Dataset(
            "D3", "E1", size_gb=1.0, residency_regions=frozenset({"east", "west"})))
        svc.fail_site("E1")
        with self.assertRaises(PlacementImpossibleError) as ctx:
            svc.submit_group(GroupSpec("g1", "T1", [
                TaskSpec("t1", gpus=2, duration_slots=1, inputs=("D3",), deadline=4),
            ]))
        codes = {r[0] for c in ctx.exception.exclusions["candidates"]
                 for r in c["hard_rejections"]}
        self.assertIn(expl.DATA_UNREACHABLE, codes)

    def test_power_tier_reduces_effective_capacity(self) -> None:
        svc = build_service()
        # W1 功率上限降到 20kW（等效 4 GPU），E1/E2 容量预先占满
        svc.topology.sites["W1"].power_capacity_kw = 20.0
        svc.state.capacity.hold_gpu("E1", 0, 12, 8, "seed:E1")
        svc.state.capacity.hold_gpu("E2", 0, 12, 8, "seed:E2")
        with self.assertRaises(PlacementImpossibleError) as ctx:
            svc.submit_group(GroupSpec("g1", "T1", [
                TaskSpec("t1", gpus=6, duration_slots=1, inputs=("D1",), deadline=4),
            ]))
        reports = {c["site_id"]: [r[0] for r in c["hard_rejections"]]
                   for c in ctx.exception.exclusions["candidates"]}
        self.assertIn(expl.SITE_POWER_CAPACITY, reports["W1"])

    def test_deadline_impossible_returns_DEADLINE(self) -> None:
        svc = build_service()
        with self.assertRaises(PlacementImpossibleError) as ctx:
            svc.submit_group(GroupSpec("g1", "T1", [
                TaskSpec("t1", gpus=2, duration_slots=5, inputs=("D1",), deadline=2),
            ]))
        codes = {r[0] for c in ctx.exception.exclusions["candidates"]
                 for r in c["hard_rejections"]}
        self.assertIn(expl.DEADLINE, codes)

    def test_transfer_budget_is_hard_constraint(self) -> None:
        svc = build_service()
        # D4 只有东站点副本但允许西区驻留：去 W1 必须跨区传输
        from compute_network_scheduler.models import Dataset
        svc.topology.add_dataset(Dataset(
            "D4", "E1", size_gb=6.0,
            residency_regions=frozenset({"east", "west"}),
            replica_sites=frozenset({"E2"}),
        ))
        # 东站点容量占满，只剩 W1：6GB 经 10Gbps 跨区链路 5 槽传完，时限内可行
        svc.state.capacity.hold_gpu("E1", 0, 12, 8, "seed:E1")
        svc.state.capacity.hold_gpu("E2", 0, 12, 8, "seed:E2")
        with self.assertRaises(PlacementImpossibleError) as ctx:
            svc.submit_group(GroupSpec("g1", "T1", [
                TaskSpec("t1", gpus=2, duration_slots=1, inputs=("D4",), deadline=12),
            ], transfer_budget_gb=5.0))
        codes = {r[0] for c in ctx.exception.exclusions["candidates"]
                 for r in c["hard_rejections"]}
        self.assertIn(expl.TRANSFER_BUDGET, codes)

    def test_failed_site_is_hard_excluded(self) -> None:
        svc = build_service()
        svc.fail_site("W1")
        svc.submit_group(GroupSpec("g1", "T1", [
            TaskSpec("t1", gpus=2, duration_slots=1, inputs=("D1",), deadline=4),
        ]))
        report = svc.explain_last_decision("g1")
        blocked = report["hard_exclusions"].get("t1", [])
        self.assertTrue(any(c["site_id"] == "W1" and
                            any(r["code"] == expl.SITE_FAILED for r in c["reasons"])
                            for c in blocked))


class SoftGoalTests(unittest.TestCase):
    def test_low_energy_tier_site_chosen_when_capacity_identical(self) -> None:
        svc = build_service()
        svc.submit_group(GroupSpec("g1", "T1", [
            TaskSpec("t1", gpus=2, duration_slots=2, inputs=("D1",)),
        ]))
        chosen = svc.get_group("g1").tasks["t1"].placement.site_id
        # D1 在 E1/E2/W1 都有副本，E1 能耗档位最低
        self.assertEqual(chosen, "E1")

    def test_local_data_avoids_transfer(self) -> None:
        svc = build_service()
        # D5 在 E1/E2 有副本、W1 无副本，但允许西区驻留
        from compute_network_scheduler.models import Dataset
        svc.topology.add_dataset(Dataset(
            "D5", "E1", size_gb=10.0,
            residency_regions=frozenset({"east", "west"}),
            replica_sites=frozenset({"E2"}),
        ))
        # 让 E1 与 E2 的能耗档位相同
        svc.topology.sites["E2"].energy_tier = svc.topology.sites["E1"].energy_tier
        svc.submit_group(GroupSpec("g1", "T1", [
            TaskSpec("t1", gpus=2, duration_slots=1, inputs=("D5",)),
        ]))
        report = svc.explain_last_decision("g1")
        for row in report["soft_ranking"]["t1"]:
            if row["site_id"] in ("E1", "E2"):
                self.assertEqual(row["soft"]["transfer_gb"], 0.0)
            if row["site_id"] == "W1":
                self.assertGreater(row["soft"]["transfer_gb"], 0.0)

    def test_soft_weights_participate_in_ranking(self) -> None:
        svc = build_service()
        # 仅保留一个候选可行时，解释中应携带全部软目标分项与权重
        svc.submit_group(GroupSpec("g1", "T1", [
            TaskSpec("t1", gpus=2, duration_slots=1, inputs=("D1",)),
        ]))
        report = svc.explain_last_decision("g1")
        self.assertEqual(set(report["soft_weights"]),
                         {"energy", "transfer", "quota", "slack"})
        row = report["soft_ranking"]["t1"][0]
        self.assertIsNotNone(row["soft"])
        self.assertIn("energy", row["soft"])

    def test_chosen_flag_unique_per_task(self) -> None:
        svc = build_service()
        svc.submit_group(GroupSpec("g1", "T1", [
            TaskSpec("t1", gpus=2, duration_slots=1, inputs=("D1",)),
        ]))
        report = svc.explain_last_decision("g1")
        chosen = [r for r in report["soft_ranking"]["t1"] if r["chosen"]]
        self.assertEqual(len(chosen), 1)


if __name__ == "__main__":
    unittest.main()
