"""JSON 仓储与服务重启后状态恢复的测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from compute_network_scheduler.enums import GroupState, TaskState
from compute_network_scheduler.repository import JsonRepository
from compute_network_scheduler.service import SchedulerService

from tests._helpers import build_service
from tests.test_lifecycle import group, task


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "state.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _open(self) -> SchedulerService:
        return SchedulerService(repository=JsonRepository(self.path))

    def test_restore_after_confirm(self) -> None:
        svc = build_service(repo=JsonRepository(self.path))
        svc.submit_group(group("g1", [task("a", inputs=("D1",))]))
        svc.reserve("g1")
        svc.confirm("g1")
        # 重新打开：时钟、占用、任务状态全部恢复
        svc2 = self._open()
        self.assertEqual(svc2.clock.now, 0)
        g = svc2.get_group("g1")
        self.assertEqual(g.state, GroupState.CONFIRMED)
        self.assertEqual(g.tasks["a"].state, TaskState.CONFIRMED)
        # 继续推进时间直至完成
        svc2.advance(3)
        self.assertEqual(svc2.get_group("g1").state, GroupState.COMPLETED)

    def test_restore_mid_run_and_failover(self) -> None:
        svc = build_service(repo=JsonRepository(self.path))
        svc.submit_group(group("g1", [
            task("etl", gpus=6, duration_slots=3, inputs=("D1",), deadline=12),
        ], ttl=3))
        svc.reserve("g1")
        svc.confirm("g1")
        svc.advance(1)
        svc.fail_site("E1")
        svc.migrate_group("g1")
        svc.advance(2)

        svc2 = self._open()
        self.assertEqual(svc2.clock.now, 3)
        rt = svc2.get_group("g1").tasks["etl"]
        self.assertEqual(rt.attempt, 2)
        self.assertIn(rt.state, (TaskState.RUNNING, TaskState.CONFIRMED))
        svc2.advance(3)
        self.assertEqual(svc2.get_group("g1").state, GroupState.COMPLETED)

    def test_quota_and_capacity_survive_restart(self) -> None:
        svc = build_service(repo=JsonRepository(self.path))
        svc.submit_group(group("g1", [task("a", gpus=4, duration_slots=2, inputs=("D1",))]))
        svc.reserve("g1")
        used = svc.tenant_quota_view("T1")["used"]
        svc2 = self._open()
        self.assertEqual(svc2.tenant_quota_view("T1")["used"], used)
        site = svc2.get_group("g1").tasks["a"].placement.site_id
        used_gpu = sum(svc2.state.capacity.gpu_used(site, s) for s in range(0, 3))
        self.assertEqual(used_gpu, 8)  # 4 GPU × 2 槽

    def test_decisions_persisted(self) -> None:
        svc = build_service(repo=JsonRepository(self.path))
        svc.submit_group(group("g1", [task("a", inputs=("D1",))]))
        decisions_before = list(svc.state.decisions)
        svc2 = self._open()
        self.assertEqual(list(svc2.state.decisions), decisions_before)
        view = svc2.explain_last_decision("g1")
        self.assertTrue(view["feasible"])


if __name__ == "__main__":
    unittest.main()
