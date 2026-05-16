# hjs-path-finding

A Rust implementation of the **Haeupler–Jiang–Saranurak (HJS) 2025** algorithm for
single-source shortest paths (SSSP) in directed graphs with negative integer weights,
together with Goldberg's practical bit-scaling algorithm, classical Bellman-Ford, and
Dijkstra as supporting subroutines.

**Paper:** _Near-Optimal Deterministic Single-Source Shortest Paths in Directed Graphs_,
Haeupler, Jiang, and Saranurak (2025).

---

## Algorithms

### `sssp` — HJS 2025 (Theorem 1.1)

```text
O(m log⁸ n · log(nW))
```

The main entry point for general-purpose SSSP with arbitrary integer weights. Applies the
BCF23 weight-scaling reduction (Lemma 5.2) to progressively reveal one bit of each weight
per phase, building a sequence of potential-adjusted graphs. Each phase is solved via
`restricted_sssp`, which drives the recursive `k_sssp` subroutine:

- **Base case** (`k ≤ log⁶ n` negative edges allowed): `sssp_few_negative` (Lemma 5.5)
  alternates Dijkstra passes with Bellman-Ford fix-up rounds.
- **Recursive case**: constructs a *d-path cover* of the non-negative sub-graph via
  `path_cover`, prunes to within-SCC edges (Lemma 5.4 premise), solves recursively
  with `k/2`, then assembles `G″` (x = 2λ tiered copies) and calls `sssp_cross_scc`.

Returns `None` if the graph contains a negative-weight cycle.

---

### `goldberg` — Goldberg's bit-scaling algorithm

```text
O(m √n log(nW))
```

Practical scaling algorithm that reveals one bit of each weight per phase, using
`sssp_minus_one` (the O(m√n) SSSP subroutine for weights in `{−1, 0, …, n}`) to repair
potentials within each phase. Competitive with HJS at benchmark sizes and simpler to reason
about. Returns `None` on a negative-weight cycle.

---

### `bellman_ford` — Bellman-Ford / Moore

```text
O(m n)
```

Runs n−1 relaxation rounds over all edges. Reliable baseline for small graphs or when
code simplicity matters. Returns `None` on a negative-weight cycle.

---

### `dijkstra` — Dijkstra with a binary heap

```text
O((m + n) log n)
```

Binary-heap Dijkstra with lazy deletion. **Requires all edge weights to be non-negative.**
Used internally as the fast solver once potentials have been computed; also the right choice
when you know your graph has no negative weights.

---

### `sssp_hjs_forced` — HJS with forced recursion

Variant of `sssp` that forces the full HJS recursive path-cover machinery to fire at every
level regardless of the negative-edge count. Useful for testing and profiling the recursive
case in isolation. In practice `sssp` is always faster because it falls back to
`sssp_few_negative` at small `k`.

---

## When to use which algorithm

| Situation | Recommended |
|-----------|-------------|
| All edge weights ≥ 0 | `dijkstra` |
| Tiny graph (n < ~50) or debugging | `bellman_ford` |
| General negative weights, practical use | `goldberg` or `sssp` |
| Need the best theoretical complexity | `sssp` (HJS) |
| Profiling / testing the recursive HJS case | `sssp_hjs_forced` |

In practice `sssp` and `goldberg` perform almost identically at the sizes benchmarked here
(up to n = 800). Both are vastly better than `bellman_ford` once the graph becomes even
slightly dense (m ≫ √n).

---

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
hjs-path-finding = { path = "." }
```

Enable optional Rayon parallelism:

```toml
[dependencies]
hjs-path-finding = { path = ".", features = ["parallel"] }
```

Basic example:

```rust,no_run
use hjs_path_finding::graph::{Graph, NodeId};
use hjs_path_finding::sssp::{sssp, INF};

// Build a small graph with a negative edge.
let mut g = Graph::with_capacity(4, 4);
g.add_nodes(4);
g.add_edge(NodeId(0), NodeId(1),  5);
g.add_edge(NodeId(1), NodeId(2), -3);
g.add_edge(NodeId(0), NodeId(2),  8);
g.add_edge(NodeId(2), NodeId(3),  2);

let dist = sssp(&g, NodeId(0)).expect("no negative cycle");

assert_eq!(dist[0],  0);   // source
assert_eq!(dist[1],  5);
assert_eq!(dist[2],  2);   // via 0→1→2: 5 + (−3) = 2
assert_eq!(dist[3],  4);   // via 0→1→2→3
```

---

## Parallelism

Enable with `--features parallel` (adds [Rayon](https://docs.rs/rayon) as a dependency).

Two adaptive thresholds gate the parallel paths:

| Threshold | Value | Guards |
|-----------|-------|--------|
| `PAR_BF_MIN_EDGES` | 4 096 edges | Bellman-Ford relaxation rounds |
| `PAR_GRAPH_MIN_EDGES` | 65 536 items | All graph-construction edge loops |

Below each threshold the code falls back to the serial path. On macOS, Rayon's thread-pool
wakeup costs roughly 50–100 µs per `into_par_iter()` call; below the thresholds this fixed
overhead would dominate the actual computation. At benchmark sizes (m ≤ 3 200) no parallel
path is taken, so `--features parallel` produces timing-identical output to serial builds.
The parallel paths become active for large graphs (m ≥ 65 536 edges, roughly n ≥ 10 000
for sparse inputs).

---

## Benchmarks

Measured on Apple M-series hardware with `[profile.bench] lto = "fat" codegen-units = 1`.
Run with:

```sh
cargo bench                        # serial
cargo bench --features parallel    # with Rayon (thresholds apply, no difference at these sizes)
```

### Sparse graph — random directed, ~4 edges per node, weights ∈ [−10, 10]

| n   | m      | HJS       | Goldberg  | Bellman-Ford |
|-----|--------|-----------|-----------|--------------|
|  50 |   ~200 |  16.4 µs  |  16.7 µs  |   7.0 µs     |
| 100 |   ~400 |  44.0 µs  |  43.9 µs  |  32.7 µs     |
| 200 |   ~800 |   130 µs  |   127 µs  |   129 µs     |
| 400 | ~1 600 |   362 µs  |   361 µs  |   531 µs     |
| 800 | ~3 200 |  1.03 ms  |  1.04 ms  |  2.43 ms     |

HJS-forced (forces full recursion at every level) is included in benches but omitted above:
it is 8–25× slower than HJS/Goldberg and intended only for algorithm research.

### Path graph — linear chain 0→1→…→(n−1), weights ∈ [−5, 5], no negative cycle

| n    | HJS      | Goldberg  | Bellman-Ford |
|------|----------|-----------|--------------|
|  200 |  74.5 µs |  74.1 µs  |    585 ns    |
|  500 |   268 µs |   269 µs  |   1.39 µs    |
| 1000 |   832 µs |   829 µs  |   2.74 µs    |

Path graphs have at most one negative edge in any path, so Bellman-Ford terminates in
O(m) time. HJS and Goldberg are orders of magnitude slower because they apply scaling
phases regardless of negative-edge structure.

### Grid graph — r×c DAG-like grid, weights ∈ [−3, 3]

| n (r×c)       | HJS      | Goldberg  | Bellman-Ford |
|---------------|----------|-----------|--------------|
| 100  (10×10)  |  17.8 µs |  18.2 µs  |    505 ns    |
| 225  (15×15)  |  42.6 µs |  43.0 µs  |   1.10 µs    |
| 400  (20×20)  |  83.6 µs |  84.8 µs  |   1.98 µs    |

Grid graphs are nearly-DAG structures where negative edges seldom form cycles. Bellman-Ford
is again extremely fast. HJS and Goldberg scale with m rather than the cycle structure.

---

## Crate structure

| Module | Contents |
|--------|----------|
| `graph` | Arena-backed directed weighted graph (`Graph`, `NodeId`, `EdgeId`) |
| `sssp` | All SSSP algorithms (public API) |
| `path_cover` | d-path cover construction (Theorem 4.5) |
| `projection` | Projection / preimage data structures used by path cover |

---

## License

MIT OR Apache-2.0
