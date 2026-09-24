"""命令行接口：拓扑管理、作业组提交、安置、确认、取消、虚拟时间推进、
故障注入、决策解释与场景复现。

运行数据默认保存在用户目录，不写入源码目录；可用 --db 或环境变量
CNS_DB 指定其他位置。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

from ..application.services import SchedulerService
from ..domain.constraints import UsageView, tenant_transfer_usage
from ..domain.errors import DomainError
from ..domain.models import JobSpec, JoinSpec, State
from ..infrastructure.ports_impl import ManualClock, SequentialIds, StateEventSink
from ..infrastructure.store import JsonStateStore
from .scenarios import SCENARIOS, build_demo_topology

DEFAULT_DB = os.environ.get(
    "CNS_DB", os.path.join(os.path.expanduser("~"), ".compute_network_scheduler", "state.json")
)


def _service_from(store: JsonStateStore) -> tuple[SchedulerService, State]:
    state = store.load()
    clock = ManualClock(state.now_seconds)
    svc = SchedulerService(state, clock, SequentialIds(state), StateEventSink(state, clock))
    return svc, state


def _run_mutating(args: argparse.Namespace, fn: Callable[[SchedulerService], Any]) -> Any:
    store = JsonStateStore(args.db)
    svc, state = _service_from(store)
    result = fn(svc)
    store.save(state)
    return result


def _run_readonly(args: argparse.Namespace, fn: Callable[[SchedulerService], Any]) -> Any:
    store = JsonStateStore(args.db)
    svc, _ = _service_from(store)
    return fn(svc)


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 各子命令
# ---------------------------------------------------------------------------


def cmd_init_demo(args: argparse.Namespace) -> None:
    store = JsonStateStore(args.db)
    if store.exists() and not args.force:
        raise DomainError("STATE_EXISTS", "状态文件已存在，使用 --force 重置")
    state = State()
    clock = ManualClock(0)
    svc = SchedulerService(state, clock, SequentialIds(state), StateEventSink(state, clock))
    build_demo_topology(svc)
    store.save(state)
    print(f"演示拓扑已初始化 -> {args.db}")
    print("站点: site-east-1(档位1,10单位) site-east-2(档位2,8单位) site-west-1(档位3,12单位)")
    print("租户: tenant-a(并发10,传输200GB) tenant-b(并发6,传输60GB)")
    print("数据: ds-east(60GB@site-east-1,可驻留三站) ds-west(30GB@site-west-1,仅本站)")


def cmd_add_park(args: argparse.Namespace) -> None:
    _run_mutating(args, lambda s: s.add_park(args.id, args.name, args.energy_cap))
    print(f"园区 {args.id} 已创建")


def cmd_add_site(args: argparse.Namespace) -> None:
    _run_mutating(
        args,
        lambda s: s.add_site(args.id, args.name, args.region, args.park, args.energy_tier, args.units, args.kwh),
    )
    print(f"站点 {args.id} 已创建")


def cmd_add_link(args: argparse.Namespace) -> None:
    _run_mutating(
        args, lambda s: s.add_link(args.id, args.site_a, args.site_b, args.bandwidth, args.max_transfers)
    )
    print(f"链路 {args.id} 已创建")


def cmd_add_tenant(args: argparse.Namespace) -> None:
    _run_mutating(args, lambda s: s.add_tenant(args.id, args.name, args.units, args.transfer_budget))
    print(f"租户 {args.id} 已创建")


def cmd_add_dataset(args: argparse.Namespace) -> None:
    _run_mutating(
        args,
        lambda s: s.add_dataset(args.id, args.name, args.size, args.location, args.allowed),
    )
    print(f"数据集 {args.id} 已创建")


def cmd_submit(args: argparse.Namespace) -> None:
    with open(args.file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    tenant = args.tenant or payload.get("tenant")
    if not tenant:
        raise DomainError("TENANT_REQUIRED", "必须通过 --tenant 或文件中的 tenant 字段指定租户")
    specs = [JobSpec.from_dict(j) for j in payload["jobs"]]
    group = _run_mutating(args, lambda s: s.submit_group(tenant, specs))
    print(f"作业组 {group.id} 已提交，包含 {len(group.job_ids)} 个作业")
    for job_id in group.job_ids:
        print(f"  {job_id}")


def cmd_place(args: argparse.Namespace) -> None:
    def work(svc: SchedulerService) -> list[str]:
        reports = svc.place_job(args.job) if args.job else svc.place_ready()
        lines = []
        for r in reports:
            job_key = svc.state.jobs[r.job_id].key
            if r.chosen_site_id:
                lines.append(f"{r.job_id}({job_key}) 子任务 {r.task_id} -> {r.chosen_site_id} [{r.reservation_id}]")
            else:
                lines.append(f"{r.job_id}({job_key}) 子任务 {r.task_id} 无可行站点（决策 {r.id}，可用 explain 查看）")
        return lines

    for line in _run_mutating(args, work):
        print(line)


def cmd_confirm(args: argparse.Namespace) -> None:
    def work(svc: SchedulerService) -> dict[str, Any]:
        if args.job:
            svc.confirm_job(args.job)
            return {"confirmed": [args.job], "failed": []}
        return svc.confirm_all()

    result = _run_mutating(args, work)
    print(f"已确认作业: {', '.join(result['confirmed']) if result['confirmed'] else '(无)'}")
    for f in result["failed"]:
        print(f"确认失败: {f['job_id']} [{f['code']}] {f['message']}")


def cmd_cancel(args: argparse.Namespace) -> None:
    _run_mutating(args, lambda s: s.cancel_job(args.job, args.reason))
    print(f"作业 {args.job} 已取消")


def cmd_tick(args: argparse.Namespace) -> None:
    def work(svc: SchedulerService) -> str:
        svc.tick(args.seconds)
        slot = svc.state.now_slot
        return f"虚拟时间: {svc.state.now_seconds} 秒（时段 {slot}）"

    print(_run_mutating(args, work))


def cmd_fail_site(args: argparse.Namespace) -> None:
    report = _run_mutating(args, lambda s: s.fail_site(args.site))
    print(f"站点 {args.site} 已失效，受控迁移结果：")
    if not report.outcomes:
        print("  (无受影响作业)")
    for o in report.outcomes:
        print(f"  {o.job_id}: {o.outcome} {o.new_site_ids} {o.detail}")


def cmd_recover_site(args: argparse.Namespace) -> None:
    _run_mutating(args, lambda s: s.recover_site(args.site))
    print(f"站点 {args.site} 已恢复")


def cmd_fail_task(args: argparse.Namespace) -> None:
    _run_mutating(args, lambda s: s.fail_task(args.task, args.reason))
    print(f"子任务 {args.task} 已注入失败")


def cmd_status(args: argparse.Namespace) -> None:
    def work(svc: SchedulerService) -> dict[str, Any]:
        st = svc.state
        usage = UsageView(st)
        sites = []
        for site in sorted(st.sites.values(), key=lambda s: s.id):
            used = usage.compute[site.id][st.now_slot]
            park = st.parks[site.park_id]
            sites.append(
                {
                    "site_id": site.id,
                    "status": site.status.value,
                    "energy_tier": site.energy_tier,
                    "compute_used_now": used,
                    "compute_cap": site.compute_units_per_slot,
                    "park_energy_used_now": round(usage.energy[site.park_id][st.now_slot], 3),
                    "park_energy_cap": park.energy_cap_per_slot,
                }
            )
        tenants = []
        for t in sorted(st.tenants.values(), key=lambda x: x.id):
            consumed, held = tenant_transfer_usage(st, t.id)
            tenants.append(
                {
                    "tenant_id": t.id,
                    "units_used_now": usage.tenant[t.id][st.now_slot],
                    "units_cap": t.max_concurrent_units,
                    "transfer_consumed_gb": round(consumed, 3),
                    "transfer_held_gb": round(held, 3),
                    "transfer_budget_gb": t.transfer_budget_gb,
                }
            )
        jobs = [
            {"job_id": j.id, "key": j.key, "tenant": j.tenant_id, "state": j.state.value}
            for j in sorted(st.jobs.values(), key=lambda j: j.id)
        ]
        return {
            "now_seconds": st.now_seconds,
            "now_slot": st.now_slot,
            "sites": sites,
            "tenants": tenants,
            "jobs": jobs,
        }

    _print_json(_run_readonly(args, work))


def cmd_jobs(args: argparse.Namespace) -> None:
    def work(svc: SchedulerService) -> Any:
        if args.group:
            return svc.group_status(args.group)
        return [
            {
                "job_id": j.id,
                "key": j.key,
                "group": j.group_id,
                "state": j.state.value,
                "tasks": [
                    {"task_id": t, "state": svc.state.tasks[t].state.value, "site": svc.state.tasks[t].site_id}
                    for t in j.task_ids
                ],
            }
            for j in sorted(svc.state.jobs.values(), key=lambda j: j.id)
        ]

    _print_json(_run_readonly(args, work))


def cmd_explain(args: argparse.Namespace) -> None:
    def work(svc: SchedulerService) -> list[dict[str, Any]]:
        reports = (
            [svc.explain_decision(args.decision)] if args.decision else svc.explain_job(args.job)
        )
        out = []
        for r in reports:
            out.append(
                {
                    "decision_id": r.id,
                    "job_id": r.job_id,
                    "task_id": r.task_id,
                    "attempt": r.attempt,
                    "chosen_site": r.chosen_site_id,
                    "reservation": r.reservation_id,
                    "note": r.note,
                    "candidates": [
                        {
                            "site_id": c.site_id,
                            "feasible": c.feasible,
                            "hard_violations": c.hard_violations,
                            "soft_scores": c.soft_scores,
                            "total_score": c.total_score,
                            "planned_slots": [c.planned_start_slot, c.planned_end_slot]
                            if c.planned_start_slot is not None
                            else None,
                        }
                        for c in r.candidates
                    ],
                }
            )
        return out

    _print_json(_run_readonly(args, work))


def cmd_events(args: argparse.Namespace) -> None:
    def work(svc: SchedulerService) -> list[dict[str, Any]]:
        return [
            {"seq": e.seq, "at": e.at_seconds, "kind": e.kind, "message": e.message, "data": e.data}
            for e in svc.state.events
            if e.seq > args.since
        ]

    _print_json(_run_readonly(args, work))


def cmd_scenario(args: argparse.Namespace) -> None:
    if args.name not in SCENARIOS:
        raise DomainError("UNKNOWN_SCENARIO", f"未知场景 {args.name}，可选: {sorted(SCENARIOS)}")
    lines, _ = SCENARIOS[args.name]()
    for line in lines:
        print(line)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compute-network-scheduler",
        description="跨区域算网作业安置服务命令行",
    )
    p.add_argument("--db", default=DEFAULT_DB, help="状态文件路径（默认 %(default)s）")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init-demo", help="初始化演示拓扑")
    sp.add_argument("--force", action="store_true", help="重置已有状态")
    sp.set_defaults(func=cmd_init_demo)

    sp = sub.add_parser("add-park", help="创建园区")
    sp.add_argument("--id", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--energy-cap", type=float, required=True, help="每时段能耗上限(千瓦时)")
    sp.set_defaults(func=cmd_add_park)

    sp = sub.add_parser("add-site", help="创建站点")
    sp.add_argument("--id", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--region", required=True)
    sp.add_argument("--park", required=True)
    sp.add_argument("--energy-tier", type=int, required=True, help="能耗档位，越小越绿色")
    sp.add_argument("--units", type=int, required=True, help="每时段算力")
    sp.add_argument("--kwh", type=float, required=True, help="每算力单位每时段能耗")
    sp.set_defaults(func=cmd_add_site)

    sp = sub.add_parser("add-link", help="创建站点间链路")
    sp.add_argument("--id", required=True)
    sp.add_argument("--site-a", required=True)
    sp.add_argument("--site-b", required=True)
    sp.add_argument("--bandwidth", type=float, required=True, help="带宽 GB/s")
    sp.add_argument("--max-transfers", type=int, required=True, help="最大并发传输数")
    sp.set_defaults(func=cmd_add_link)

    sp = sub.add_parser("add-tenant", help="创建租户")
    sp.add_argument("--id", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--units", type=int, required=True, help="并发算力额度")
    sp.add_argument("--transfer-budget", type=float, required=True, help="网络传输预算 GB")
    sp.set_defaults(func=cmd_add_tenant)

    sp = sub.add_parser("add-dataset", help="登记数据集")
    sp.add_argument("--id", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--size", type=float, required=True, help="大小 GB")
    sp.add_argument("--location", required=True, help="所在站点")
    sp.add_argument("--allowed", nargs="+", required=True, help="允许驻留处理的站点")
    sp.set_defaults(func=cmd_add_dataset)

    sp = sub.add_parser("submit", help="提交带依赖的作业组（JSON 文件）")
    sp.add_argument("--file", required=True)
    sp.add_argument("--tenant", default=None)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("place", help="对 READY 作业执行安置")
    sp.add_argument("--job", default=None, help="仅安置指定作业")
    sp.set_defaults(func=cmd_place)

    sp = sub.add_parser("confirm", help="确认预留")
    sp.add_argument("--job", default=None, help="仅确认指定作业；缺省确认全部")
    sp.set_defaults(func=cmd_confirm)

    sp = sub.add_parser("cancel", help="执行前取消作业")
    sp.add_argument("--job", required=True)
    sp.add_argument("--reason", default="")
    sp.set_defaults(func=cmd_cancel)

    sp = sub.add_parser("tick", help="推进虚拟时间")
    sp.add_argument("--seconds", type=int, required=True)
    sp.set_defaults(func=cmd_tick)

    sp = sub.add_parser("fail-site", help="站点失效并执行受控迁移")
    sp.add_argument("--site", required=True)
    sp.set_defaults(func=cmd_fail_site)

    sp = sub.add_parser("recover-site", help="恢复站点")
    sp.add_argument("--site", required=True)
    sp.set_defaults(func=cmd_recover_site)

    sp = sub.add_parser("fail-task", help="注入子任务执行失败")
    sp.add_argument("--task", required=True)
    sp.add_argument("--reason", default="注入故障")
    sp.set_defaults(func=cmd_fail_task)

    sp = sub.add_parser("status", help="总览：时间、站点、租户、作业")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("jobs", help="作业列表或作业组详情")
    sp.add_argument("--group", default=None)
    sp.set_defaults(func=cmd_jobs)

    sp = sub.add_parser("explain", help="解释决策：硬约束排除与软目标排序")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--job", default=None)
    g.add_argument("--decision", default=None)
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("events", help="事件日志")
    sp.add_argument("--since", type=int, default=0, help="仅显示序号大于该值的事件")
    sp.set_defaults(func=cmd_events)

    sp = sub.add_parser("scenario", help="运行复现场景（内存态，不影响 --db）")
    sp.add_argument("name", choices=sorted(SCENARIOS))
    sp.set_defaults(func=cmd_scenario)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except DomainError as e:
        print(f"错误[{e.code}]: {e.message}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
