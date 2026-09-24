"""命令行入口。

用法：

    python3 -m compute_network_scheduler.cli reset
    python3 -m compute_network_scheduler.cli demo quota-race|ttl-expiry|failover|partial-failure
    python3 -m compute_network_scheduler.cli advance 3
    python3 -m compute_network_scheduler.cli groups
    python3 -m compute_network_scheduler.cli explain <group_id>
    python3 -m compute_network_scheduler.cli reserve|confirm|cancel|migrate <group_id>
    python3 -m compute_network_scheduler.cli fail-site <site_id>
    python3 -m compute_network_scheduler.cli fail-task <group_id> <task_id>
    python3 -m compute_network_scheduler.cli quota <tenant_id>
    python3 -m compute_network_scheduler.cli serve [--port 8080]

状态文件默认位于运行目录下的 .runtime/state.json（已在 .gitignore 排除），
可用环境变量 CNS_STATE 覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .enums import EnergyTier
from .errors import SchedulerError
from .models import Dataset, GroupSpec, Link, Site, TaskSpec, Tenant
from .repository import JsonRepository
from .service import SchedulerService

DEFAULT_STATE = os.environ.get(
    "CNS_STATE", os.path.join(".runtime", "state.json")
)


# ----------------------------------------------------------------------
# 演示拓扑
# ----------------------------------------------------------------------
def seed_demo(service: SchedulerService) -> None:
    """固定的三园区两链路演示拓扑，保证场景可重复。"""
    topo = service.topology
    topo.add_site(Site("park-east-1", "east", gpu_capacity=8, power_capacity_kw=40,
                       kw_per_gpu=5, energy_tier=EnergyTier.LOW, cost_per_gpu_slot=2.0))
    topo.add_site(Site("park-east-2", "east", gpu_capacity=8, power_capacity_kw=48,
                       kw_per_gpu=5, energy_tier=EnergyTier.MEDIUM, cost_per_gpu_slot=2.0))
    topo.add_site(Site("park-west-1", "west", gpu_capacity=8, power_capacity_kw=40,
                       kw_per_gpu=5, energy_tier=EnergyTier.HIGH, cost_per_gpu_slot=3.0))
    topo.add_link(Link("link-east", "park-east-1", "park-east-2",
                       bandwidth_gbps=20.0, cost_per_gb=0.5))
    topo.add_link(Link("link-cross", "park-east-2", "park-west-1",
                       bandwidth_gbps=8.0, cost_per_gb=0.8))
    topo.add_dataset(Dataset(
        "ds-ledger", "park-east-1", size_gb=20.0,
        residency_regions=frozenset({"east", "west"}),
        replica_sites=frozenset({"park-east-2", "park-west-1"}),
    ))
    topo.add_dataset(Dataset(
        "ds-east-only", "park-east-1", size_gb=10.0,
        residency_regions=frozenset({"east"}),
        replica_sites=frozenset({"park-east-2"}),
    ))
    topo.add_tenant(Tenant("bank", quota_limit=50.0))


def fresh_service(path: str = DEFAULT_STATE, *, seed: bool = True) -> SchedulerService:
    if os.path.exists(path):
        os.unlink(path)
    service = SchedulerService(repository=JsonRepository(path))
    if seed:
        seed_demo(service)
    return service


def open_service(path: str = DEFAULT_STATE) -> SchedulerService:
    service = SchedulerService(repository=JsonRepository(path))
    if not service.topology.sites:
        seed_demo(service)
    return service


def _line(title: str) -> None:
    print(f"\n=== {title} ===")


def _quota(service: SchedulerService) -> float:
    return service.tenant_quota_view("bank")["used"]


# ----------------------------------------------------------------------
# 复现场景
# ----------------------------------------------------------------------
def scenario_quota_race(service: SchedulerService) -> None:
    _line("场景一：额度竞争（每作业需 36 额度，租户上限 50）")
    for gid in ("batch-A", "batch-B"):
        service.submit_group(GroupSpec(gid, "bank", [
            TaskSpec("etl", gpus=6, duration_slots=3, inputs=("ds-ledger",),
                     deadline=12),
        ], reservation_ttl_slots=3))
    rsv = service.reserve("batch-A")
    print(f"[t={service.clock.now}] batch-A 预留成功 {rsv.reservation_id}，"
          f"额度占用 {_quota(service)}")
    # 客户端超时重试 reserve：必须幂等返回同一预留，不重复扣减
    rsv_retry = service.reserve("batch-A")
    print(f"[重试] reserve 幂等返回同一预留：{rsv_retry.reservation_id == rsv.reservation_id}，"
          f"额度仍为 {_quota(service)}")
    try:
        service.reserve("batch-B")
    except SchedulerError as exc:
        excl = exc.exclusions
        print(f"batch-B 被硬约束拒绝：{exc}")
        for cand in excl["candidates"]:
            codes = ", ".join(r[0] for r in cand["hard_rejections"])
            print(f"  - 候选 {cand['site_id']}：{codes}")
    print(f"额度占用保持 {_quota(service)}，拒绝方未产生任何扣减")


def scenario_ttl_expiry(service: SchedulerService) -> None:
    _line("场景二：预留 TTL 超时自动释放")
    service.submit_group(GroupSpec("batch-TTL", "bank", [
        TaskSpec("etl", gpus=4, duration_slots=2, inputs=("ds-ledger",), deadline=10),
    ], reservation_ttl_slots=2))
    service.reserve("batch-TTL")
    print(f"[t={service.clock.now}] 已预留，额度 {_quota(service)}")
    service.advance(2)
    g = service.get_group("batch-TTL")
    print(f"[t={service.clock.now}] 超时后：组状态={g.state.value}，"
          f"任务状态={g.tasks['etl'].state.value}，额度={_quota(service)}")
    service.reserve("batch-TTL")
    print(f"[t={service.clock.now}] 重新预留成功，额度恢复为 {_quota(service)}（无重复扣减）")
    service.confirm("batch-TTL")
    service.advance(4)
    print(f"确认并推进后：组状态={service.get_group('batch-TTL').state.value}")


def scenario_failover(service: SchedulerService) -> None:
    _line("场景三：站点失效后的受控迁移")
    service.submit_group(GroupSpec("batch-FO", "bank", [
        TaskSpec("etl", gpus=6, duration_slots=3, inputs=("ds-ledger",), deadline=12),
    ], reservation_ttl_slots=3))
    service.reserve("batch-FO")
    service.confirm("batch-FO")
    service.advance(1)
    p0 = service.get_group("batch-FO").tasks["etl"].placement
    print(f"[t={service.clock.now}] 任务正在 {p0.site_id} 执行")
    out = service.fail_site("park-east-1")
    print(f"站点失效，受影响：{out['affected']}")
    service.migrate_group("batch-FO")
    g = service.get_group("batch-FO")
    p1 = g.tasks["etl"].placement
    print(f"受控迁移：{p1.migrated_from} -> {p1.site_id}，attempt={g.tasks['etl'].attempt}，"
          f"数据来源副本={list(p1.sources.values())}")
    entries = service.tenant_quota_view("bank")["entries"]
    print("额度账本行：", {k: ("已退还" if v["refunded"] else "生效")
                           for k, v in entries.items()})
    service.advance(5)
    g = service.get_group("batch-FO")
    print(f"[t={service.clock.now}] 组状态={g.state.value}，"
          f"任务状态={g.tasks['etl'].state.value}")


def scenario_partial_failure(service: SchedulerService) -> None:
    _line("场景四：部分子任务失败，屏障永不放行")
    service.submit_group(GroupSpec("batch-PF", "bank", [
        TaskSpec("extract", gpus=2, duration_slots=3, inputs=("ds-ledger",)),
        TaskSpec("load", gpus=2, duration_slots=1, depends_on=("extract",), deadline=10),
    ], reservation_ttl_slots=3))
    service.reserve("batch-PF")
    service.confirm("batch-PF")
    service.advance(1)
    print(f"[t={service.clock.now}] extract 执行中，注入永久故障")
    service.inject_task_failure("batch-PF", "extract")
    g = service.get_group("batch-PF")
    print(f"组状态={g.state.value}；"
          f"extract={g.tasks['extract'].state.value}，"
          f"load={g.tasks['load'].state.value}")
    print("屏障语义：extract 未成功，load 即便未运行也不得完成；"
          "没有任何子任务进入 completed")
    try:
        service.migrate_group("batch-PF")
    except SchedulerError as exc:
        print(f"迁移被正确拒绝：{exc}")
    print("最近一次决策的硬约束/软目标可通过 explain 子命令查询")


SCENARIOS = {
    "quota-race": scenario_quota_race,
    "ttl-expiry": scenario_ttl_expiry,
    "failover": scenario_failover,
    "partial-failure": scenario_partial_failure,
}


# ----------------------------------------------------------------------
# 命令
# ----------------------------------------------------------------------
def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compute_network_scheduler",
                                     description="跨区域算网作业安置服务")
    parser.add_argument("--state", default=DEFAULT_STATE, help="状态文件路径")
    state_parent = argparse.ArgumentParser(add_help=False)
    state_parent.add_argument("--state", default=DEFAULT_STATE, help="状态文件路径")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("reset", parents=[state_parent], help="清空并重新播种演示拓扑")
    p_demo = sub.add_parser("demo", parents=[state_parent], help="运行复现场景")
    p_demo.add_argument("scenario", choices=sorted(SCENARIOS))

    sub.add_parser("groups", parents=[state_parent], help="列出作业组")
    p_group = sub.add_parser("group", parents=[state_parent], help="作业组详情")
    p_group.add_argument("group_id")
    p_explain = sub.add_parser("explain", parents=[state_parent],
                               help="查询最近决策的硬约束排除与软目标排序")
    p_explain.add_argument("group_id")
    p_dec = sub.add_parser("decision", parents=[state_parent], help="查询指定决策")
    p_dec.add_argument("decision_id")
    p_quota = sub.add_parser("quota", parents=[state_parent], help="租户额度视图")
    p_quota.add_argument("tenant_id", nargs="?", default="bank")
    sub.add_parser("capacity", parents=[state_parent], help="站点容量与占用快照")

    p_adv = sub.add_parser("advance", parents=[state_parent], help="推进虚拟时间")
    p_adv.add_argument("slots", type=int)
    for name in ("reserve", "confirm", "cancel", "migrate"):
        p = sub.add_parser(name, parents=[state_parent])
        p.add_argument("group_id")
    p_fail = sub.add_parser("fail-site", parents=[state_parent])
    p_fail.add_argument("site_id")
    p_rec = sub.add_parser("recover-site", parents=[state_parent])
    p_rec.add_argument("site_id")
    p_ft = sub.add_parser("fail-task", parents=[state_parent])
    p_ft.add_argument("group_id")
    p_ft.add_argument("task_id")

    p_serve = sub.add_parser("serve", parents=[state_parent], help="启动 HTTP 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = args.cmd

    if cmd == "reset":
        fresh_service(args.state)
        print(f"已在 {args.state} 重置并播种演示拓扑")
        return 0
    if cmd == "demo":
        service = fresh_service(args.state)
        SCENARIOS[args.scenario](service)
        return 0
    if cmd == "serve":
        from .api import serve
        httpd = serve(args.host, args.port, args.state)
        print(f"HTTP 服务监听 http://{args.host}:{args.port}（状态文件 {args.state}）")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    service = open_service(args.state)
    try:
        if cmd == "groups":
            _print([
                {"group_id": g.group_id, "state": g.state.value,
                 "tasks": {tid: rt.state.value for tid, rt in g.tasks.items()}}
                for g in service.list_groups()
            ])
        elif cmd == "group":
            _print(service.get_group(args.group_id).to_dict())
        elif cmd == "explain":
            _print(service.explain_last_decision(args.group_id))
        elif cmd == "decision":
            _print(service.get_decision(args.decision_id).to_dict())
        elif cmd == "quota":
            _print(service.tenant_quota_view(args.tenant_id))
        elif cmd == "capacity":
            _print(service.capacity_view())
        elif cmd == "advance":
            _print(service.advance(args.slots))
        elif cmd == "reserve":
            _print(service.reserve(args.group_id).to_dict())
        elif cmd == "confirm":
            _print(service.confirm(args.group_id).to_dict())
        elif cmd == "cancel":
            _print(service.cancel_group(args.group_id).to_dict())
        elif cmd == "migrate":
            _print(service.migrate_group(args.group_id).to_dict())
        elif cmd == "fail-site":
            _print(service.fail_site(args.site_id))
        elif cmd == "recover-site":
            _print(service.recover_site(args.site_id))
        elif cmd == "fail-task":
            _print(service.inject_task_failure(args.group_id, args.task_id))
    except SchedulerError as exc:
        print(f"错误[{type(exc).__name__}] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
