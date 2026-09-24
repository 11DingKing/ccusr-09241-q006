"""作业组依赖：一次提交、DAG 校验、依赖门控。"""

import unittest

from compute_network_scheduler.domain.errors import DomainError
from compute_network_scheduler.domain.models import JobSpec, JobState
from helpers import assert_invariants, make_service


class DependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = make_service()

    def test_group_submitted_atomically_with_dag(self) -> None:
        svc = self.svc
        group = svc.submit_group(
            "t1",
            [
                JobSpec(key="extract", dataset_id="d1", compute_units=2, duration_slots=1, deadline_slot=10),
                JobSpec(
                    key="train", dataset_id="d1", compute_units=2, duration_slots=1,
                    deadline_slot=10, depends_on=["extract"],
                ),
                JobSpec(
                    key="eval", dataset_id="d1", compute_units=2, duration_slots=1,
                    deadline_slot=10, depends_on=["train"],
                ),
            ],
        )
        self.assertEqual(len(group.job_ids), 3)
        jobs = {j.key: j for j in svc.state.jobs.values()}
        self.assertEqual(jobs["extract"].state, JobState.READY)
        self.assertEqual(jobs["train"].state, JobState.PENDING)
        self.assertEqual(jobs["eval"].state, JobState.PENDING)

    def test_cycle_rejected(self) -> None:
        with self.assertRaises(DomainError) as ctx:
            self.svc.submit_group(
                "t1",
                [
                    JobSpec(key="a", dataset_id="d1", compute_units=1, duration_slots=1,
                            deadline_slot=5, depends_on=["b"]),
                    JobSpec(key="b", dataset_id="d1", compute_units=1, duration_slots=1,
                            deadline_slot=5, depends_on=["a"]),
                ],
            )
        self.assertEqual(ctx.exception.code, "DEP_CYCLE")
        self.assertEqual(len(self.svc.state.jobs), 0, "校验失败不得残留作业")

    def test_self_dependency_rejected(self) -> None:
        with self.assertRaises(DomainError) as ctx:
            self.svc.submit_group(
                "t1",
                [JobSpec(key="a", dataset_id="d1", compute_units=1, duration_slots=1,
                         deadline_slot=5, depends_on=["a"])],
            )
        self.assertEqual(ctx.exception.code, "DEP_CYCLE")

    def test_unknown_dependency_rejected(self) -> None:
        with self.assertRaises(DomainError) as ctx:
            self.svc.submit_group(
                "t1",
                [JobSpec(key="a", dataset_id="d1", compute_units=1, duration_slots=1,
                         deadline_slot=5, depends_on=["ghost"])],
            )
        self.assertEqual(ctx.exception.code, "DEP_NOT_FOUND")

    def test_dependent_unblocks_only_after_dependency_succeeds(self) -> None:
        svc = self.svc
        svc.submit_group(
            "t1",
            [
                JobSpec(key="up", dataset_id="d1", compute_units=2, duration_slots=2, deadline_slot=10),
                JobSpec(key="down", dataset_id="d1", compute_units=2, duration_slots=2,
                        deadline_slot=15, depends_on=["up"]),
            ],
        )
        jobs = {j.key: j for j in svc.state.jobs.values()}
        svc.place_ready()
        self.assertEqual(jobs["down"].state, JobState.PENDING)
        svc.confirm_all()
        svc.tick(300)
        self.assertEqual(jobs["down"].state, JobState.PENDING)  # 上游尚未完成
        svc.tick(300)  # 上游完成
        self.assertEqual(jobs["up"].state, JobState.SUCCEEDED)
        self.assertEqual(jobs["down"].state, JobState.READY)
        svc.place_ready()
        svc.confirm_all()
        svc.tick(600)
        self.assertEqual(jobs["down"].state, JobState.SUCCEEDED)
        assert_invariants(svc.state)

    def test_dependency_failure_blocks_dependent(self) -> None:
        svc = self.svc
        svc.submit_group(
            "t1",
            [
                JobSpec(key="up", dataset_id="d1", compute_units=2, duration_slots=2, deadline_slot=10),
                JobSpec(key="down", dataset_id="d1", compute_units=2, duration_slots=2,
                        deadline_slot=15, depends_on=["up"]),
            ],
        )
        jobs = {j.key: j for j in svc.state.jobs.values()}
        svc.place_ready()
        svc.confirm_all()
        svc.tick(300)
        up_task = svc.state.tasks[jobs["up"].task_ids[0]]
        svc.fail_task(up_task.id, "注入失败")
        self.assertEqual(jobs["up"].state, JobState.FAILED)
        svc.tick(1)  # 触发依赖刷新
        self.assertEqual(jobs["down"].state, JobState.BLOCKED)
        assert_invariants(svc.state)


if __name__ == "__main__":
    unittest.main()
