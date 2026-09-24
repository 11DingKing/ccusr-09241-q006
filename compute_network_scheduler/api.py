"""HTTP 接口层（标准库实现，无第三方依赖）。

路由：

    GET  /health
    POST /admin/sites|links|datasets|tenants      资源注册
    POST /groups                                  提交作业组
    GET  /groups                                  作业组列表
    GET  /groups/{id}                             作业组详情
    POST /groups/{id}/reserve|confirm|cancel
    POST /groups/{id}/migrate
    POST /tasks/{gid}/{tid}/fail                  子任务执行失败注入
    POST /sites/{id}/fail|recover
    POST /clock/advance        {"slots": n}
    GET  /quota/{tenant}
    GET  /decisions/{id}
    GET  /groups/{id}/explain
    GET  /capacity
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .enums import EnergyTier
from .errors import SchedulerError
from .models import Dataset, GroupSpec, Link, Site, TaskSpec, Tenant
from .repository import JsonRepository
from .service import SchedulerService


def _json_error(exc: Exception) -> tuple[int, dict]:
    code = type(exc).__name__
    status = 409
    if code in ("NotFoundError", "UnknownSiteError"):
        status = 404
    elif code == "ValidationError":
        status = 422
    elif code == "PlacementImpossibleError":
        status = 409
    payload = {"error": code, "message": str(exc)}
    if hasattr(exc, "exclusions") and exc.exclusions:
        payload["exclusions"] = exc.exclusions
    return status, payload


class SchedulerHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state_path: str | None) -> None:
        self.state_path = state_path
        repo = JsonRepository(state_path) if state_path else None
        self.service = SchedulerService(repository=repo)
        self.lock = threading.RLock()
        super().__init__(address, _make_handler())


def _make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server: SchedulerHttpServer

        def log_message(self, fmt: str, *args: Any) -> None:  # 静默
            return

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _serve(self, fn: Callable[[SchedulerService], Any]) -> None:
            with self.server.lock:
                service = self.server.service
                try:
                    result = fn(service)
                    self._send(200, result if result is not None else {"ok": True})
                except SchedulerError as exc:
                    status, payload = _json_error(exc)
                    self._send(status, payload)

        # ---- 路由 ------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                self._send(200, {"status": "ok"})
                return
            m = re.fullmatch(r"/groups/([^/]+)/explain", path)
            if m:
                self._serve(lambda s: s.explain_last_decision(m.group(1)))
                return
            m = re.fullmatch(r"/groups/([^/]+)", path)
            if m:
                self._serve(lambda s: s.get_group(m.group(1)).to_dict())
                return
            if path == "/groups":
                self._serve(lambda s: [g.to_dict() for g in s.list_groups()])
                return
            m = re.fullmatch(r"/quota/([^/]+)", path)
            if m:
                self._serve(lambda s: s.tenant_quota_view(m.group(1)))
                return
            m = re.fullmatch(r"/decisions/([^/]+)", path)
            if m:
                self._serve(lambda s: s.get_decision(m.group(1)).to_dict())
                return
            if path == "/capacity":
                self._serve(lambda s: s.capacity_view())
                return
            self._send(404, {"error": "NotFound", "message": path})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            body = self._read_json()

            def run(fn: Callable[[SchedulerService], Any]) -> None:
                self._serve(fn)

            if path == "/admin/sites":
                def op(s):
                    s.topology.add_site(Site(
                        site_id=body["site_id"], region=body["region"],
                        gpu_capacity=int(body["gpu_capacity"]),
                        power_capacity_kw=float(body["power_capacity_kw"]),
                        kw_per_gpu=float(body["kw_per_gpu"]),
                        energy_tier=EnergyTier(body.get("energy_tier", "medium")),
                        cost_per_gpu_slot=float(body["cost_per_gpu_slot"]),
                    ))
                    return {"ok": True}
                run(op)
            elif path == "/admin/links":
                def op(s):
                    s.topology.add_link(Link(
                        link_id=body["link_id"], site_a=body["site_a"], site_b=body["site_b"],
                        bandwidth_gbps=float(body["bandwidth_gbps"]),
                        cost_per_gb=float(body.get("cost_per_gb", 0.0)),
                    ))
                    return {"ok": True}
                run(op)
            elif path == "/admin/datasets":
                def op(s):
                    s.topology.add_dataset(Dataset(
                        dataset_id=body["dataset_id"], site_id=body["site_id"],
                        size_gb=float(body["size_gb"]),
                        residency_regions=frozenset(body.get("residency_regions", [])),
                        replica_sites=frozenset(body.get("replica_sites", [])),
                    ))
                    return {"ok": True}
                run(op)
            elif path == "/admin/tenants":
                def op(s):
                    s.topology.add_tenant(Tenant(
                        tenant_id=body["tenant_id"], quota_limit=float(body["quota_limit"])))
                    return {"ok": True}
                run(op)
            elif path == "/groups":
                def op(s):
                    tasks = [TaskSpec(
                        task_id=t["task_id"], gpus=int(t["gpus"]),
                        duration_slots=int(t["duration_slots"]),
                        inputs=tuple(t.get("inputs", [])),
                        depends_on=tuple(t.get("depends_on", [])),
                        earliest_start=int(t.get("earliest_start", 0)),
                        deadline=t.get("deadline"),
                    ) for t in body["tasks"]]
                    spec = GroupSpec(
                        group_id=body["group_id"], tenant_id=body["tenant_id"],
                        tasks=tasks,
                        reservation_ttl_slots=int(body.get("reservation_ttl_slots", 2)),
                        transfer_budget_gb=body.get("transfer_budget_gb"),
                    )
                    return s.submit_group(spec).to_dict()
                run(op)
            elif path == "/clock/advance":
                run(lambda s: s.advance(int(body.get("slots", 1))))
            else:
                m = re.fullmatch(r"/groups/([^/]+)/(reserve|confirm|cancel|migrate)", path)
                if m:
                    gid, action = m.group(1), m.group(2)
                    run({
                        "reserve": lambda s: s.reserve(gid).to_dict(),
                        "confirm": lambda s: s.confirm(gid).to_dict(),
                        "cancel": lambda s: s.cancel_group(gid).to_dict(),
                        "migrate": lambda s: s.migrate_group(gid).to_dict(),
                    }[action])
                    return
                m = re.fullmatch(r"/sites/([^/]+)/(fail|recover)", path)
                if m:
                    sid, action = m.group(1), m.group(2)
                    run(lambda s: s.fail_site(sid) if action == "fail"
                        else s.recover_site(sid))
                    return
                m = re.fullmatch(r"/tasks/([^/]+)/([^/]+)/fail", path)
                if m:
                    run(lambda s: s.inject_task_failure(m.group(1), m.group(2)))
                    return
                self._send(404, {"error": "NotFound", "message": path})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080,
          state_path: str | None = None) -> ThreadingHTTPServer:
    httpd = SchedulerHttpServer((host, port), state_path)
    return httpd
