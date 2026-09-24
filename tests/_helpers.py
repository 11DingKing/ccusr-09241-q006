"""测试共享的拓扑构造辅助。"""

from __future__ import annotations

from compute_network_scheduler.enums import EnergyTier
from compute_network_scheduler.models import Dataset, Link, Site, Tenant
from compute_network_scheduler.service import SchedulerService


def build_service(quota_limit: float = 100.0, *, repo=None) -> SchedulerService:
    """三园区（东2西1）、两链路、两数据集（含多副本）、单租户的标准测试拓扑。"""
    svc = SchedulerService(repository=repo)
    topo = svc.topology
    topo.add_site(Site("E1", "east", gpu_capacity=8, power_capacity_kw=40,
                       kw_per_gpu=5, energy_tier=EnergyTier.LOW, cost_per_gpu_slot=2.0))
    topo.add_site(Site("E2", "east", gpu_capacity=8, power_capacity_kw=48,
                       kw_per_gpu=5, energy_tier=EnergyTier.MEDIUM, cost_per_gpu_slot=2.0))
    topo.add_site(Site("W1", "west", gpu_capacity=8, power_capacity_kw=40,
                       kw_per_gpu=5, energy_tier=EnergyTier.HIGH, cost_per_gpu_slot=3.0))
    topo.add_link(Link("LE", "E1", "E2", bandwidth_gbps=10.0, cost_per_gb=0.5))
    topo.add_link(Link("LW", "E2", "W1", bandwidth_gbps=10.0, cost_per_gb=0.8))
    topo.add_dataset(Dataset(
        "D1", "E1", size_gb=10.0,
        residency_regions=frozenset({"east", "west"}),
        replica_sites=frozenset({"E2", "W1"}),
    ))
    topo.add_dataset(Dataset(
        "D2", "E1", size_gb=10.0,
        residency_regions=frozenset({"east"}),
        replica_sites=frozenset({"E2"}),
    ))
    topo.add_tenant(Tenant("T1", quota_limit=quota_limit))
    return svc
