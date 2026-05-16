"""
Unit tests for the hjs_sssp Python bindings.

Run from the project root after building the extension:

    uv run maturin develop --features python
    uv run python -m unittest discover -s tests -v

Or with pytest (if installed):

    uv run pytest tests/ -v
"""

import unittest

import networkx as nx

import hjs_sssp

INF = float("inf")

# Algorithms under test — same interface, exercised identically.
ALGORITHMS = [hjs_sssp.sssp, hjs_sssp.goldberg, hjs_sssp.bellman_ford]


def _run(algo, g, source, **kw):
    """Call *algo* and return the dist dict (raises on unexpected None)."""
    result = algo(g, source, **kw)
    return result


class TestSmallGraph(unittest.TestCase):
    """Basic correctness on a hand-crafted 4-node graph with a negative edge."""

    def setUp(self):
        self.g = nx.DiGraph()
        self.g.add_edge(0, 1, weight=5)
        self.g.add_edge(1, 2, weight=-3)
        self.g.add_edge(0, 2, weight=8)
        self.g.add_edge(2, 3, weight=2)

    def _check(self, algo):
        dist = _run(algo, self.g, 0)
        self.assertIsNotNone(dist)
        self.assertEqual(dist[0], 0)
        self.assertEqual(dist[1], 5)
        self.assertEqual(dist[2], 2)  # 0→1→2: 5 + (−3) = 2
        self.assertEqual(dist[3], 4)  # via node 2

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


class TestNegativeCycle(unittest.TestCase):
    """Graphs containing a negative-weight cycle must return None."""

    def setUp(self):
        self.g = nx.DiGraph()
        self.g.add_edge(0, 1, weight=1)
        self.g.add_edge(1, 2, weight=-4)
        self.g.add_edge(2, 0, weight=2)  # cycle weight = 1 − 4 + 2 = −1

    def _check(self, algo):
        result = algo(self.g, 0)
        self.assertIsNone(
            result, f"{algo.__name__} should return None on negative cycle"
        )

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


class TestUnreachableNodes(unittest.TestCase):
    """Nodes not reachable from source should map to None."""

    def setUp(self):
        self.g = nx.DiGraph()
        self.g.add_edge(0, 1, weight=3)
        self.g.add_node(2)  # isolated — unreachable from 0

    def _check(self, algo):
        dist = _run(algo, self.g, 0)
        self.assertIsNotNone(dist)
        self.assertEqual(dist[0], 0)
        self.assertEqual(dist[1], 3)
        self.assertIsNone(dist[2], f"node 2 should be unreachable")

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


class TestSingleNode(unittest.TestCase):
    """A graph with a single node and no edges."""

    def setUp(self):
        self.g = nx.DiGraph()
        self.g.add_node(42)

    def _check(self, algo):
        dist = _run(algo, self.g, 42)
        self.assertIsNotNone(dist)
        self.assertEqual(dist[42], 0)

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


class TestNonIntegerNodes(unittest.TestCase):
    """Nodes can be arbitrary hashable Python objects (strings, tuples, etc.)."""

    def setUp(self):
        self.g = nx.DiGraph()
        self.g.add_edge("a", "b", weight=2)
        self.g.add_edge("b", "c", weight=-1)
        self.g.add_edge("a", "c", weight=10)

    def _check(self, algo):
        dist = _run(algo, self.g, "a")
        self.assertIsNotNone(dist)
        self.assertEqual(dist["a"], 0)
        self.assertEqual(dist["b"], 2)
        self.assertEqual(dist["c"], 1)  # "a"→"b"→"c": 2 + (−1) = 1

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


class TestCustomWeightAttribute(unittest.TestCase):
    """The `weight` kwarg selects an alternative edge attribute."""

    def setUp(self):
        self.g = nx.DiGraph()
        self.g.add_edge(0, 1, cost=7)
        self.g.add_edge(1, 2, cost=-2)

    def _check(self, algo):
        dist = _run(algo, self.g, 0, weight="cost")
        self.assertIsNotNone(dist)
        self.assertEqual(dist[0], 0)
        self.assertEqual(dist[1], 7)
        self.assertEqual(dist[2], 5)

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


class TestDefaultWeight(unittest.TestCase):
    """Edges missing the weight attribute use `default_weight`."""

    def setUp(self):
        self.g = nx.DiGraph()
        self.g.add_edge(0, 1)  # no weight attribute
        self.g.add_edge(1, 2)

    def _check(self, algo):
        dist = _run(algo, self.g, 0, default_weight=3)
        self.assertIsNotNone(dist)
        self.assertEqual(dist[0], 0)
        self.assertEqual(dist[1], 3)
        self.assertEqual(dist[2], 6)

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


class TestMatchesNetworkX(unittest.TestCase):
    """
    Cross-validate against networkx.single_source_bellman_ford_path_length
    on a random-ish 20-node graph with mixed weights.
    """

    def setUp(self):
        import random

        rng = random.Random(0)
        self.g = nx.DiGraph()
        n = 20
        for u in range(n):
            for _ in range(4):
                v = rng.randrange(n)
                if u != v:
                    self.g.add_edge(u, v, weight=rng.randint(-5, 10))

    def _check(self, algo):
        source = 0
        try:
            nx_dist = dict(nx.single_source_bellman_ford_path_length(self.g, source))
        except nx.NetworkXUnbounded:
            # NetworkX detected a negative cycle — our algorithms should also return None.
            result = algo(self.g, source)
            self.assertIsNone(result)
            return

        dist = _run(algo, self.g, source)
        self.assertIsNotNone(dist)
        for node, expected in nx_dist.items():
            self.assertEqual(
                dist[node],
                expected,
                f"{algo.__name__}: dist[{node}] = {dist[node]}, expected {expected}",
            )

    def test_sssp(self):
        self._check(hjs_sssp.sssp)

    def test_goldberg(self):
        self._check(hjs_sssp.goldberg)

    def test_bellman_ford(self):
        self._check(hjs_sssp.bellman_ford)


if __name__ == "__main__":
    unittest.main()
