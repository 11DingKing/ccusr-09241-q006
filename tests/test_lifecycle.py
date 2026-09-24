"""生命周期与业务不变量测试：预留/TTL/取消/迁移/屏障/额度幂等。"""

from __future__ import annotations

import unittest

from compute_network_scheduler.enums import GroupState, TaskState
from compute_network_scheduler.errors import (
    PlacementImpossibleError,
    ReservationExpiredError,
    StateConflictError,
)
from compute_network_scheduler.models import GroupSpec, TaskSpec

from tests._helpers import build_service


def task(tid: str, **kw) -> TaskSpec:
    kw.setdefault("gpus", 2)
    kw.setdefault("duration_slots", 1)
    return TaskSpec(tid, **kw)


def group(gid: str, tasks, ttl: int = 3, **kw) -> GroupSpec:
    return GroupSpec(gid, "T1", tasks, reservation_ttl_slots=ttl, **kw)


class QuotaIdempotencyTests(unittest.TestCase):
    def test_repeated_reserve_does_not_double_debit(self) -> None:
        svc = build_service(quota_limit=100)
        svc.submit_group(group("g1", [task("a", inputs=("D1",))]))
        svc.reserve("g1")
        used1 = svc.tenant_quota_view("T1")["used"]
        # 模拟客户端重试：连续重复预留必须返回同一预留且额度不变
        rsv2 = svc.reserve("g1")
        rsv3 = svc.reserve("g1")
        self.assertEqual(rsv2.reservation_id, rsv3.reservation_id)
        self.assertEqual(svc.tenant_quota_view("T1")["used"], used1)
        self.assertEqual(len(svc.tenant_quota_view("T1")["entries"]), 1)

    def test_failed_reserve_debits_nothing(self) -> None:
        svc = build_service(quota_limit=40)
        # g1 提交时规划可行（成本 36 < 40）；g0 先预留占走 8，g1 预留时重规划必败
        svc.submit_group(group("g0", [task("a", duration_slots=2, inputs=("D1",))]))
        svc.submit_group(group("g1", [task("a", gpus=6, duration_slots=3, inputs=("D1",))]))
        svc.reserve("g0")
        before = svc.tenant_quota_view("T1")["used"]
        self.assertEqual(before, 8.0)
        with self.assertRaises(PlacementImpossibleError):
            svc.reserve("g1")
        # 失败后额度账上没有 g1 的任何残留，且可反复重试
        self.assertEqual(svc.tenant_quota_view("T1")["used"], 8.0)
        with self.assertRaises(PlacementImpossibleError):
            svc.reserve("g1")
        self.assertEqual(svc.tenant_quota_view("T1")["used"], 8.0)

    def test_cancel_refunds_quota_and_capacity(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [task("a", inputs=("D1",))]))
        svc.reserve("g1")
        self.assertGreater(svc.tenant_quota_view("T1")["used"], 0)
        svc.cancel_group("g1")
        self.assertEqual(svc.tenant_quota_view("T1")["used"], 0.0)
        self.assertEqual(svc.get_group("g1").state, GroupState.CANCELLED)
        # 容量占用全部释放
        gpu = svc.state.capacity
        self.assertTrue(all(
            sum(bucket.values()) == 0
            for slots in gpu._gpu.values() for bucket in slots.values()
        ))

    def test_cancel_after_execution_started_rejected(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [task("a", duration_slots=2, inputs=("D1",))]))
        svc.reserve("g1")
        svc.confirm("g1")
        svc.advance(1)
        with self.assertRaises(StateConflictError):
            svc.cancel_group("g1")

    def test_ttl_expiry_releases_and_re_reserve_debits_once(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [task("a", inputs=("D1",))], ttl=2))
        svc.reserve("g1")
        used = svc.tenant_quota_view("T1")["used"]
        svc.advance(2)
        self.assertEqual(svc.tenant_quota_view("T1")["used"], 0.0)
        self.assertIsNone(svc.get_group("g1").reservation)
        # 重新预留：同一幂等键重新生效，账上仍只有一笔
        svc.reserve("g1")
        view = svc.tenant_quota_view("T1")
        self.assertEqual(view["used"], used)
        active = [k for k, v in view["entries"].items() if not v["refunded"]]
        self.assertEqual(active, ["g1:a:a1"])

    def test_confirm_after_expiry_rejected(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [task("a", inputs=("D1",))], ttl=1))
        svc.reserve("g1")
        svc.advance(1)
        with self.assertRaises(ReservationExpiredError):
            svc.confirm("g1")


class ReservationLifecycleTests(unittest.TestCase):
    def test_cannot_confirm_without_reserve(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [task("a", inputs=("D1",))]))
        with self.assertRaises(ReservationExpiredError):
            svc.confirm("g1")

    def test_quota_competition_blocks_second_group(self) -> None:
        svc = build_service(quota_limit=40)
        # 每个作业成本 6*3*2 = 36，只够一个
        svc.submit_group(group("g1", [task("a", gpus=6, duration_slots=3, inputs=("D1",))]))
        svc.submit_group(group("g2", [task("a", gpus=6, duration_slots=3, inputs=("D1",))]))
        svc.reserve("g1")
        with self.assertRaises(PlacementImpossibleError):
            svc.reserve("g2")
        self.assertEqual(svc.tenant_quota_view("T1")["used"], 36.0)
        # g1 取消后 g2 立刻可预留
        svc.cancel_group("g1")
        svc.reserve("g2")
        self.assertEqual(svc.tenant_quota_view("T1")["used"], 36.0)


class DependencyAndBarrierTests(unittest.TestCase):
    def test_downstream_starts_only_after_upstream_finishes(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [
            task("a", duration_slots=2, inputs=("D1",)),
            task("b", duration_slots=1, depends_on=("a",)),
        ]))
        svc.reserve("g1")
        svc.confirm("g1")
        svc.advance(1)
        g = svc.get_group("g1")
        self.assertEqual(g.tasks["a"].state, TaskState.RUNNING)
        self.assertIn(g.tasks["b"].state, (TaskState.CONFIRMED,))
        pa, pb = g.tasks["a"].placement, g.tasks["b"].placement
        self.assertGreaterEqual(pb.start_slot, pa.finish_slot)
        svc.advance(3)
        g = svc.get_group("g1")
        self.assertEqual(g.state, GroupState.COMPLETED)
        self.assertTrue(all(rt.state == TaskState.COMPLETED for rt in g.tasks.values()))

    def test_barrier_blocks_completion_when_subtask_fails(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [
            task("a", duration_slots=3, inputs=("D1",)),
            task("b", duration_slots=1, depends_on=("a",)),
        ]))
        svc.reserve("g1")
        svc.confirm("g1")
        svc.advance(1)
        svc.inject_task_failure("g1", "a")
        g = svc.get_group("g1")
        self.assertEqual(g.state, GroupState.FAILED)
        # 任何子任务都不得进入完成态
        for rt in g.tasks.values():
            self.assertNotEqual(rt.state, TaskState.COMPLETED)
            self.assertNotEqual(rt.state, TaskState.SUCCEEDED)

    def test_succeeded_tasks_stay_succeeded_until_barrier(self) -> None:
        """b 依赖 a 但长期无法启动时，a 成功后停在 SUCCEEDED，不能提前 COMPLETED。"""
        svc = build_service()
        # a 账本占用槽 0..1、在槽 1..2 执行；b 最早槽 2。占满槽 2..20，
        # 规划器仍可在槽 21 安置 b（提交可行），但推进 2 槽时 b 无法启动
        for sid in ("E1", "E2", "W1"):
            svc.state.capacity.hold_gpu(sid, 2, 20, 8, f"block:{sid}")
        svc.submit_group(group("g1", [
            task("a", duration_slots=2, inputs=("D1",)),
            task("b", gpus=6, duration_slots=1, depends_on=("a",)),
        ]))
        svc.reserve("g1")
        svc.confirm("g1")
        svc.advance(2)
        g = svc.get_group("g1")
        self.assertEqual(g.tasks["a"].state, TaskState.SUCCEEDED)
        self.assertEqual(g.tasks["b"].state, TaskState.CONFIRMED)
        self.assertNotEqual(g.state, GroupState.COMPLETED)


class FailoverMigrationTests(unittest.TestCase):
    def _running_group(self, svc, gid="g1"):
        svc.submit_group(group(gid, [
            task("etl", gpus=6, duration_slots=3, inputs=("D1",), deadline=12),
        ], ttl=3))
        svc.reserve(gid)
        svc.confirm(gid)
        svc.advance(1)

    def test_migration_uses_new_attempt_and_refunds_old(self) -> None:
        svc = build_service()
        self._running_group(svc)
        old_site = svc.get_group("g1").tasks["etl"].placement.site_id
        svc.fail_site(old_site)
        svc.migrate_group("g1")
        g = svc.get_group("g1")
        rt = g.tasks["etl"]
        self.assertEqual(rt.attempt, 2)
        self.assertEqual(rt.placement.migrated_from, old_site)
        self.assertNotEqual(rt.placement.site_id, old_site)
        self.assertEqual(rt.placement.attempt, 2)
        # 旧账已退、新账生效，额度恰好等于一次执行的价格
        view = svc.tenant_quota_view("T1")
        entries = view["entries"]
        self.assertTrue(entries["g1:etl:a1"]["refunded"])
        self.assertFalse(entries["g1:etl:a2"]["refunded"])
        self.assertAlmostEqual(view["used"], rt.placement.quota_cost)

    def test_migrated_task_completes_after_rerun(self) -> None:
        svc = build_service()
        self._running_group(svc)
        old_site = svc.get_group("g1").tasks["etl"].placement.site_id
        svc.fail_site(old_site)
        svc.migrate_group("g1")
        svc.advance(6)
        g = svc.get_group("g1")
        self.assertEqual(g.state, GroupState.COMPLETED)
        self.assertEqual(g.tasks["etl"].state, TaskState.COMPLETED)

    def test_migration_decision_excludes_failed_site(self) -> None:
        svc = build_service()
        self._running_group(svc)
        old_site = svc.get_group("g1").tasks["etl"].placement.site_id
        svc.fail_site(old_site)
        svc.migrate_group("g1")
        report = svc.explain_last_decision("g1")
        self.assertEqual(report["kind"], "migration")
        excluded = {c["site_id"]: [r["code"] for r in c["reasons"]]
                     for c in report["hard_exclusions"].get("etl", [])}
        self.assertIn(old_site, excluded)
        self.assertIn("SITE_FAILED", excluded[old_site])

    def test_failed_site_never_chosen_again(self) -> None:
        svc = build_service()
        self._running_group(svc)
        old_site = svc.get_group("g1").tasks["etl"].placement.site_id
        svc.fail_site(old_site)
        svc.migrate_group("g1")
        self.assertNotEqual(svc.get_group("g1").tasks["etl"].placement.site_id, old_site)

    def test_site_failure_before_confirm_releases_whole_reservation(self) -> None:
        svc = build_service()
        svc.submit_group(group("g1", [
            task("etl", gpus=6, duration_slots=2, inputs=("D1",)),
        ], ttl=5))
        svc.reserve("g1")
        target = svc.get_group("g1").tasks["etl"].placement.site_id
        svc.fail_site(target)
        # 未确认的预留整体释放：额度归零、无 HELD 预留
        self.assertEqual(svc.tenant_quota_view("T1")["used"], 0.0)
        # 受控迁移后整组可以继续走完
        svc.migrate_group("g1")
        svc.advance(6)
        self.assertEqual(svc.get_group("g1").state, GroupState.COMPLETED)

    def test_permanent_failure_cannot_migrate(self) -> None:
        svc = build_service()
        self._running_group(svc)
        svc.inject_task_failure("g1", "etl")
        with self.assertRaises(StateConflictError):
            svc.migrate_group("g1")

    def test_mid_chain_failure_replans_downstream_and_barrier_holds(self) -> None:
        """a→b→c 链中 b 的站点失效：b 与未完成的 c 一并换绑 attempt+1，
        a 保持成功；最终三者按依赖顺序越过屏障完成。"""
        svc = build_service(quota_limit=500)
        svc.submit_group(group("g1", [
            task("a", duration_slots=1, inputs=("D1",)),
            task("b", duration_slots=2, depends_on=("a",), inputs=("D1",)),
            task("c", duration_slots=1, depends_on=("b",)),
        ], ttl=5))
        svc.reserve("g1")
        svc.confirm("g1")
        svc.advance(1)  # a 成功，b 执行中
        b_site = svc.get_group("g1").tasks["b"].placement.site_id
        svc.fail_site(b_site)
        svc.migrate_group("g1")
        g = svc.get_group("g1")
        self.assertEqual(g.tasks["a"].attempt, 1)
        self.assertEqual(g.tasks["a"].state, TaskState.SUCCEEDED)
        self.assertEqual(g.tasks["b"].attempt, 2)
        self.assertEqual(g.tasks["c"].attempt, 2)
        self.assertEqual(g.tasks["b"].placement.migrated_from, b_site)
        # 旧账全部退还，新账生效
        entries = svc.tenant_quota_view("T1")["entries"]
        self.assertTrue(entries["g1:b:a1"]["refunded"])
        self.assertTrue(entries["g1:c:a1"]["refunded"])
        self.assertFalse(entries["g1:b:a2"]["refunded"])
        svc.advance(6)
        g = svc.get_group("g1")
        self.assertEqual(g.state, GroupState.COMPLETED)
        self.assertTrue(all(rt.state == TaskState.COMPLETED for rt in g.tasks.values()))

    def test_migration_history_keeps_separate_decisions(self) -> None:
        svc = build_service()
        self._running_group(svc)
        old_site = svc.get_group("g1").tasks["etl"].placement.site_id
        svc.fail_site(old_site)
        svc.migrate_group("g1")
        decisions = svc.list_decisions("g1")
        self.assertGreaterEqual(len(decisions), 3)  # 提交 + 预留重规划 + 迁移
        kinds = [d.kind for d in decisions]
        self.assertEqual(kinds.count("migration"), 1)


class CapacityLedgerTests(unittest.TestCase):
    def test_gpu_capacity_never_overcommitted(self) -> None:
        svc = build_service(quota_limit=1000)
        # 两个各需 6 GPU 的作业：站点容量 8，不能落在同槽
        svc.submit_group(group("g1", [task("a", gpus=6, duration_slots=2, inputs=("D1",))]))
        svc.submit_group(group("g2", [task("a", gpus=6, duration_slots=2, inputs=("D1",))]))
        svc.reserve("g1")
        svc.reserve("g2")
        for sid, site in svc.topology.sites.items():
            cap = svc.state.capacity.effective_gpu_capacity(site)
            for slot in range(0, 8):
                self.assertLessEqual(svc.state.capacity.gpu_used(sid, slot), cap)

    def test_bandwidth_capacity_never_overcommitted(self) -> None:
        """大数据量任务必须跨区时，两条传输在同一链路上的预约速率不超带宽。"""
        from compute_network_scheduler.models import Dataset
        svc = build_service(quota_limit=1000)
        # D6 无 W1 副本但允许西区驻留：去 W1 必经 LW
        svc.topology.add_dataset(Dataset(
            "D6", "E1", size_gb=8.0,
            residency_regions=frozenset({"east", "west"}),
            replica_sites=frozenset({"E2"}),
        ))
        # 东站点容量占满，两个任务都只能去 W1，共享 LW（10Gbps）
        for sid in ("E1", "E2"):
            svc.state.capacity.hold_gpu(sid, 0, 16, 8, f"block:{sid}")
        for gid in ("g1", "g2"):
            svc.submit_group(group(gid, [
                task("a", gpus=2, duration_slots=1, inputs=("D6",), deadline=16),
            ]))
        svc.reserve("g1")
        svc.reserve("g2")
        link = svc.topology.links["LW"]
        for slot in range(0, 16):
            self.assertLessEqual(
                svc.state.capacity.bw_used("LW", slot),
                link.bandwidth_gbps + 1e-9,
                f"链路 LW 在槽 {slot} 超带宽",
            )


if __name__ == "__main__":
    unittest.main()
