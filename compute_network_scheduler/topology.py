"""资源拓扑：站点、链路、数据集、租户的注册与网络路径计算。"""

from __future__ import annotations

import heapq

from .errors import UnknownSiteError, ValidationError
from .models import Dataset, Link, Site, Tenant


class Topology:
    """内存态资源目录，提供站点图上的路径查询。"""

    def __init__(self) -> None:
        self.sites: dict[str, Site] = {}
        self.links: dict[str, Link] = {}
        self.datasets: dict[str, Dataset] = {}
        self.tenants: dict[str, Tenant] = {}
        # site_id -> {neighbor_site_id -> link_id}
        self._adj: dict[str, dict[str, str]] = {}

    # ---- 注册 ----------------------------------------------------------
    def add_site(self, site: Site) -> None:
        if site.site_id in self.sites:
            raise ValidationError(f"站点 {site.site_id} 已存在")
        self.sites[site.site_id] = site
        self._adj[site.site_id] = {}

    def add_link(self, link: Link) -> None:
        if link.link_id in self.links:
            raise ValidationError(f"链路 {link.link_id} 已存在")
        if link.site_a not in self.sites or link.site_b not in self.sites:
            raise ValidationError(f"链路 {link.link_id} 引用了未注册的站点")
        if link.bandwidth_gbps <= 0:
            raise ValidationError("链路带宽必须为正数")
        self.links[link.link_id] = link
        self._adj[link.site_a][link.site_b] = link.link_id
        self._adj[link.site_b][link.site_a] = link.link_id

    def add_dataset(self, dataset: Dataset) -> None:
        if dataset.dataset_id in self.datasets:
            raise ValidationError(f"数据集 {dataset.dataset_id} 已存在")
        if dataset.site_id not in self.sites:
            raise ValidationError(f"数据集 {dataset.dataset_id} 位于未注册站点 {dataset.site_id}")
        self.datasets[dataset.dataset_id] = dataset

    def add_tenant(self, tenant: Tenant) -> None:
        if tenant.tenant_id in self.tenants:
            raise ValidationError(f"租户 {tenant.tenant_id} 已存在")
        self.tenants[tenant.tenant_id] = tenant

    def require_site(self, site_id: str) -> Site:
        site = self.sites.get(site_id)
        if site is None:
            raise UnknownSiteError(site_id)
        return site

    def online_sites(self) -> list[Site]:
        return [s for s in self.sites.values() if s.status.value == "online"]

    # ---- 路径 ----------------------------------------------------------
    def widest_path(
        self, src: str, dst: str, failed_sites: frozenset[str] = frozenset()
    ) -> tuple[list[Link], float] | None:
        """返回瓶颈带宽最大的路径（最宽路径，Dijkstra 变体）。

        :return: (按序链路列表, 瓶颈带宽)；不可达时返回 None。
        """
        if src == dst:
            return [], float("inf")
        if src not in self.sites or dst not in self.sites:
            return None
        best: dict[str, float] = {src: float("inf")}
        prev: dict[str, tuple[str, str]] = {}
        # 堆元素：(-bottleneck, site)，源点的“负无穷”用 None 标记
        heap: list[tuple[float, str]] = [(float("-inf"), src)]
        while heap:
            neg_b, node = heapq.heappop(heap)
            bottleneck = float("inf") if neg_b == float("-inf") else -neg_b
            if node != src and bottleneck < best.get(node, -1.0):
                continue
            if node == dst:
                break
            for neighbor, link_id in self._adj.get(node, {}).items():
                if neighbor in failed_sites or node in failed_sites:
                    continue
                link = self.links[link_id]
                edge_b = min(bottleneck, link.bandwidth_gbps)
                if edge_b > best.get(neighbor, -1.0):
                    best[neighbor] = edge_b
                    prev[neighbor] = (node, link_id)
                    heapq.heappush(heap, (-edge_b, neighbor))
        if dst not in best:
            return None
        edges: list[Link] = []
        cur = dst
        while cur != src:
            node, link_id = prev[cur]
            edges.append(self.links[link_id])
            cur = node
        edges.reverse()
        return edges, best[dst]

    # ---- 序列化 --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "sites": {sid: s.to_dict() for sid, s in self.sites.items()},
            "links": {lid: l.to_dict() for lid, l in self.links.items()},
            "datasets": {did: d.to_dict() for did, d in self.datasets.items()},
            "tenants": {tid: t.to_dict() for tid, t in self.tenants.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Topology":
        topo = cls()
        for site in data.get("sites", {}).values():
            topo.add_site(Site.from_dict(site))
        for link in data.get("links", {}).values():
            topo.add_link(Link.from_dict(link))
        for dataset in data.get("datasets", {}).values():
            topo.add_dataset(Dataset.from_dict(dataset))
        for tenant in data.get("tenants", {}).values():
            topo.add_tenant(Tenant.from_dict(tenant))
        return topo
