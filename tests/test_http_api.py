"""HTTP 接口层的端到端测试（标准库 urllib，无第三方依赖）。"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from compute_network_scheduler.api import SchedulerHttpServer


def _req(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class HttpApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd = SchedulerHttpServer(("127.0.0.1", 0), None)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=2)
        self.httpd.server_close()

    def _seed(self) -> None:
        _req("POST", f"{self.base}/admin/sites", {
            "site_id": "s1", "region": "east", "gpu_capacity": 8,
            "power_capacity_kw": 40, "kw_per_gpu": 5,
            "energy_tier": "low", "cost_per_gpu_slot": 2})
        _req("POST", f"{self.base}/admin/tenants",
             {"tenant_id": "t1", "quota_limit": 100})
        _req("POST", f"{self.base}/admin/datasets", {
            "dataset_id": "d1", "site_id": "s1", "size_gb": 10,
            "residency_regions": ["east"]})

    def test_health(self) -> None:
        status, body = _req("GET", f"{self.base}/health")
        self.assertEqual((status, body["status"]), (200, "ok"))

    def test_full_flow_over_http(self) -> None:
        self._seed()
        status, body = _req("POST", f"{self.base}/groups", {
            "group_id": "g1", "tenant_id": "t1",
            "tasks": [{"task_id": "a", "gpus": 2, "duration_slots": 1,
                       "inputs": ["d1"]}],
            "reservation_ttl_slots": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "planned")

        status, body = _req("POST", f"{self.base}/groups/g1/reserve")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "held")

        status, _ = _req("POST", f"{self.base}/groups/g1/confirm")
        self.assertEqual(status, 200)

        status, body = _req("POST", f"{self.base}/clock/advance", {"slots": 2})
        self.assertEqual(status, 200)
        self.assertIn("group_completed", [e["type"] for e in body["events"]])

        status, body = _req("GET", f"{self.base}/groups/g1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "completed")

    def test_quota_failure_returns_409_with_exclusions(self) -> None:
        self._seed()
        _req("POST", f"{self.base}/groups", {
            "group_id": "g1", "tenant_id": "t1",
            "tasks": [{"task_id": "a", "gpus": 2, "duration_slots": 1,
                       "inputs": ["d1"]}]})
        _req("POST", f"{self.base}/groups/g1/reserve")
        # 租户只剩 96 额度：把额度上限调低做不到，改用超大作业
        status, body = _req("POST", f"{self.base}/groups", {
            "group_id": "g2", "tenant_id": "t1",
            "tasks": [{"task_id": "a", "gpus": 40, "duration_slots": 1,
                       "inputs": ["d1"]}]})
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "PlacementImpossibleError")
        self.assertIn("candidates", body["exclusions"])

    def test_unknown_group_returns_404(self) -> None:
        status, body = _req("GET", f"{self.base}/groups/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"], "NotFoundError")


if __name__ == "__main__":
    unittest.main()
