"""时钟、拓扑路径与最宽路径选择的单元测试。"""

from __future__ import annotations

import unittest

from compute_network_scheduler.clock import VirtualClock
from compute_network_scheduler.enums import EnergyTier
from compute_network_scheduler.models import Link, Site
from compute_network_scheduler.topology import Topology


class ClockTests(unittest.TestCase):
    def test_clock_only_moves_forward_explicitly(self) -> None:
        clock = VirtualClock()
        self.assertEqual(clock.now, 0)
        self.assertEqual(clock.advance(3), 3)
        self.assertEqual(clock.now, 3)
        with self.assertRaises(ValueError):
            clock.advance(0)
        with self.assertRaises(ValueError):
            clock.advance(-1)

    def test_clock_restore(self) -> None:
        self.assertEqual(VirtualClock.restore(7).now, 7)


class WidestPathTests(unittest.TestCase):
    def _topo(self) -> Topology:
        topo = Topology()
        for sid, region in (("A", "r1"), ("B", "r1"), ("C", "r1"), ("D", "r1")):
            topo.add_site(Site(sid, region, 8, 40, 5, EnergyTier.LOW, 1.0))
        # A-B 100, B-C 50, A-C 20, C-D 80
        topo.add_link(Link("l1", "A", "B", 100.0, 0.0))
        topo.add_link(Link("l2", "B", "C", 50.0, 0.0))
        topo.add_link(Link("l3", "A", "C", 20.0, 0.0))
        topo.add_link(Link("l4", "C", "D", 80.0, 0.0))
        return topo

    def test_widest_path_prefers_high_bottleneck(self) -> None:
        topo = self._topo()
        edges, bw = topo.widest_path("A", "C")
        self.assertEqual(bw, 50.0)  # A-B-C 瓶颈 50 优于直连 20
        self.assertEqual([e.link_id for e in edges], ["l1", "l2"])

    def test_local_path_is_infinite(self) -> None:
        topo = self._topo()
        edges, bw = topo.widest_path("A", "A")
        self.assertEqual(edges, [])
        self.assertEqual(bw, float("inf"))

    def test_unreachable_returns_none(self) -> None:
        topo = self._topo()
        topo.add_site(Site("Z", "r2", 8, 40, 5, EnergyTier.LOW, 1.0))
        self.assertIsNone(topo.widest_path("A", "Z"))

    def test_failed_site_removed_from_path(self) -> None:
        topo = self._topo()
        # B 失效后 A 到 C 只能走低带宽直连
        edges, bw = topo.widest_path("A", "C", failed_sites=frozenset({"B"}))
        self.assertEqual([e.link_id for e in edges], ["l3"])
        self.assertEqual(bw, 20.0)

    def test_failed_cut_yields_unreachable(self) -> None:
        topo = self._topo()
        # C 失效后 D 不可达
        self.assertIsNone(topo.widest_path("A", "D", failed_sites=frozenset({"C"})))


if __name__ == "__main__":
    unittest.main()
