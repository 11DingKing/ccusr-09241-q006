"""拆分作业与汇合条件：子任务只有满足汇合条件才可进入完成态。"""

import unittest

from compute_network_scheduler.domain.models import JobSpec, JobState, JoinSpec, TaskState
from helpers import assert_invariants, make_service, submit


class SplitJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = make_service()

    def _submit_split(self, splits: int, join: JoinSpec, units: int = 6, duration: int = 2):
        job = submit(
            self.svc, compute_units=units, duration_slots=duration,
            deadline_slot=20, splits=splits, join=join,
        )
        self.svc.place_ready()
        self.svc.confirm_all()
        return job

    def test_split_divides_compute_and_shards(self) -> None:
        svc = self.svc
        job = submit(svc, compute_units=7, duration_slots=2, deadline_slot=20, splits=3)
        tasks = [svc.state.tasks[t] for t in job.task_ids]
        self.assertEqual([t.compute_units for t in tasks], [3, 2, 2])
        self.assertEqual(sum(t.compute_units for t in tasks), 7)
        for t in tasks:
            self.assertAlmostEqual(t.shard_gb, 10.0)  # 30GB / 3

    def test_quorum_join_waits_for_barrier(self) -> None:
        svc = self.svc
        job = self._submit_split(3, JoinSpec(kind="quorum", k=2))
        tasks = [svc.state.tasks[t] for t in job.task_ids]
        svc.tick(300)
        self.assertTrue(all(t.state == TaskState.RUNNING for t in tasks))
        svc.tick(300)  # 全部执行完毕
        # 三个子任务同时到达汇合屏障：2/3 满足后全部进入完成态
        self.assertEqual(job.state, JobState.SUCCEEDED)
        self.assertEqual(sum(1 for t in tasks if t.state == TaskState.SUCCEEDED), 3)
        assert_invariants(svc.state)

    def test_subtask_cannot_complete_before_join_satisfied(self) -> None:
        svc = self.svc
        # 手工构造：两个子任务先后完成，quorum=2
        job = self._submit_split(2, JoinSpec(kind="quorum", k=2))
        tasks = sorted((svc.state.tasks[t] for t in job.task_ids), key=lambda t: t.id)
        svc.tick(300)
        # 强制第一个子任务先完成（直接进入 JOIN_WAIT）
        tasks[0].state = TaskState.JOIN_WAIT
        svc._evaluate_join(job)
        self.assertEqual(tasks[0].state, TaskState.JOIN_WAIT, "汇合未满足不得进入完成态")
        self.assertNotEqual(job.state, JobState.SUCCEEDED)
        # 第二个子任务完成 -> 汇合满足，两者一起进入完成态
        tasks[1].state = TaskState.JOIN_WAIT
        svc._evaluate_join(job)
        self.assertEqual(tasks[0].state, TaskState.SUCCEEDED)
        self.assertEqual(tasks[1].state, TaskState.SUCCEEDED)
        self.assertEqual(job.state, JobState.SUCCEEDED)
        assert_invariants(svc.state)

    def test_partial_failure_tolerated_by_quorum(self) -> None:
        svc = self.svc
        job = self._submit_split(3, JoinSpec(kind="quorum", k=2))
        tasks = [svc.state.tasks[t] for t in job.task_ids]
        svc.tick(300)
        svc.fail_task(tasks[0].id, "节点故障")
        self.assertEqual(job.state, JobState.RUNNING)  # 2/3 仍可能满足
        svc.tick(300)
        self.assertEqual(job.state, JobState.SUCCEEDED)
        succeeded = [t for t in tasks if t.state == TaskState.SUCCEEDED]
        self.assertEqual(len(succeeded), 2)
        self.assertEqual(tasks[0].state, TaskState.FAILED)
        assert_invariants(svc.state)

    def test_all_join_fails_on_any_subtask_failure(self) -> None:
        svc = self.svc
        job = self._submit_split(3, JoinSpec(kind="all"))
        tasks = [svc.state.tasks[t] for t in job.task_ids]
        svc.tick(300)
        svc.fail_task(tasks[1].id, "节点故障")
        self.assertEqual(job.state, JobState.FAILED)
        for t in tasks:
            self.assertNotEqual(t.state, TaskState.SUCCEEDED)
        # 资源全部释放
        assert_invariants(svc.state)

    def test_quorum_impossible_when_too_many_failures(self) -> None:
        svc = self.svc
        job = self._submit_split(3, JoinSpec(kind="quorum", k=2))
        tasks = [svc.state.tasks[t] for t in job.task_ids]
        svc.tick(300)
        svc.fail_task(tasks[0].id)
        svc.fail_task(tasks[1].id)
        self.assertEqual(job.state, JobState.FAILED)
        assert_invariants(svc.state)

    def test_split_placement_is_atomic(self) -> None:
        svc = self.svc
        # t2 额度 4：拆 2 份每份 3 单位，第二个子任务必然超额 -> 整体回滚
        svc.add_dataset("d3", "小数据", 2.0, "s1", ["s1", "s2", "s3"])
        job = submit(
            svc, tenant="t2", dataset_id="d3", compute_units=6, duration_slots=3,
            deadline_slot=3, splits=2, join=JoinSpec(kind="all"),
        )
        svc.place_ready()
        self.assertEqual(job.state, JobState.READY)
        holding = [r for r in svc.state.reservations.values() if r.status.value in ("HELD", "CONFIRMED")]
        self.assertEqual(holding, [], "拆分安置失败必须整体回滚，不得残留预留")
        assert_invariants(svc.state)


if __name__ == "__main__":
    unittest.main()
