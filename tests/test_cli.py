"""命令行端到端：虚拟时间推进复现四类场景，状态跨调用持久化。"""

import json
import os
import tempfile
import unittest

from compute_network_scheduler.interface.cli import main


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "state.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, *argv: str) -> int:
        return main(["--db", self.db, *argv])

    def submit_file(self, payload: dict) -> str:
        path = os.path.join(self.tmp.name, "jobs.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def test_full_lifecycle_via_cli(self) -> None:
        self.assertEqual(self.run_cli("init-demo"), 0)
        path = self.submit_file(
            {
                "tenant": "tenant-a",
                "jobs": [
                    {"key": "etl", "dataset_id": "ds-east", "compute_units": 4,
                     "duration_slots": 2, "deadline_slot": 10},
                    {"key": "report", "dataset_id": "ds-east", "compute_units": 2,
                     "duration_slots": 1, "deadline_slot": 15, "depends_on": ["etl"]},
                ],
            }
        )
        self.assertEqual(self.run_cli("submit", "--file", path), 0)
        self.assertEqual(self.run_cli("place"), 0)
        self.assertEqual(self.run_cli("confirm"), 0)
        self.assertEqual(self.run_cli("tick", "--seconds", "600"), 0)
        self.assertEqual(self.run_cli("place"), 0)  # 依赖就绪后安置 report
        self.assertEqual(self.run_cli("confirm"), 0)
        self.assertEqual(self.run_cli("tick", "--seconds", "600"), 0)

        # 状态跨调用持久化：重新加载校验
        from compute_network_scheduler.infrastructure.store import JsonStateStore

        state = JsonStateStore(self.db).load()
        states = {j.key: j.state.value for j in state.jobs.values()}
        self.assertEqual(states, {"etl": "SUCCEEDED", "report": "SUCCEEDED"})
        self.assertEqual(state.now_seconds, 1200)

    def test_explain_outputs_hard_and_soft_factors(self) -> None:
        self.assertEqual(self.run_cli("init-demo"), 0)
        path = self.submit_file(
            {
                "tenant": "tenant-b",  # 并发额度 6
                "jobs": [
                    {"key": "too-big", "dataset_id": "ds-east", "compute_units": 9,
                     "duration_slots": 2, "deadline_slot": 10},
                ],
            }
        )
        self.assertEqual(self.run_cli("submit", "--file", path), 0)
        self.assertEqual(self.run_cli("place"), 0)
        from compute_network_scheduler.infrastructure.store import JsonStateStore

        state = JsonStateStore(self.db).load()
        job_id = next(iter(state.jobs))
        # explain 为只读命令，直接调用服务层校验内容
        from compute_network_scheduler.interface.scenarios import make_service

        svc = make_service(state)
        reports = svc.explain_job(job_id)
        self.assertTrue(reports)
        report = reports[-1]
        self.assertIsNone(report.chosen_site_id)
        for cand in report.candidates:
            self.assertFalse(cand.feasible)
            self.assertIn("TENANT_QUOTA", cand.hard_violations)

    def test_fail_site_and_recover_via_cli(self) -> None:
        self.assertEqual(self.run_cli("init-demo"), 0)
        path = self.submit_file(
            {
                "tenant": "tenant-a",
                "jobs": [
                    {"key": "j", "dataset_id": "ds-east", "compute_units": 4,
                     "duration_slots": 2, "deadline_slot": 20},
                ],
            }
        )
        self.run_cli("submit", "--file", path)
        self.run_cli("place")
        self.run_cli("confirm")
        self.assertEqual(self.run_cli("fail-site", "--site", "site-east-1"), 0)
        self.assertEqual(self.run_cli("tick", "--seconds", "1200"), 0)
        from compute_network_scheduler.infrastructure.store import JsonStateStore

        state = JsonStateStore(self.db).load()
        job = next(iter(state.jobs.values()))
        self.assertEqual(job.state.value, "SUCCEEDED")
        task = state.tasks[job.task_ids[0]]
        self.assertNotEqual(task.site_id, "site-east-1")
        self.assertEqual(self.run_cli("recover-site", "--site", "site-east-1"), 0)

    def test_domain_error_exit_code(self) -> None:
        self.assertEqual(self.run_cli("init-demo"), 0)
        self.assertEqual(self.run_cli("confirm", "--job", "job-9999"), 2)
        self.assertEqual(self.run_cli("cancel", "--job", "job-9999"), 2)

    def test_scenarios_run_and_reproduce(self) -> None:
        for name, marker in [
            ("quota-contention", "TENANT_QUOTA"),
            ("reservation-expiry", "EXPIRED"),
            ("fault-migration", "MIGRATED"),
            ("partial-subtask-failure", "quorum"),
        ]:
            with self.subTest(scenario=name):
                import io
                from contextlib import redirect_stdout

                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main(["--db", self.db, "scenario", name])
                self.assertEqual(rc, 0)
                self.assertIn(marker, buf.getvalue())
        # 场景在内存态运行，不污染 --db
        self.assertFalse(os.path.exists(self.db))


if __name__ == "__main__":
    unittest.main()
