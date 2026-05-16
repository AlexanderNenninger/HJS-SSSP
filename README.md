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

## Building

### Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| [Rust](https://rustup.rs) ≥ 1.87 (edition 2024) | Compile the library | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| [uv](https://docs.astral.sh/uv/) | Python environment & task runner | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [maturin](https://www.maturin.rs) ≥ 1.0 | Build the PyO3 extension | installed automatically via `uv sync` |

### Rust library

```sh
# Run the full test suite (40 tests)
cargo test

# Build optimised (fat-LTO) benchmarks and run them
cargo bench

# Enable optional Rayon parallelism
cargo test --features parallel
cargo bench --features parallel

# Generate API documentation and open it
cargo doc --open
```

### Python extension (optional)

```sh
# Create the virtual environment and install dependencies (maturin, networkx)
uv sync --group dev

# Compile and install the extension into the venv (debug build, fast iteration)
uv run maturin develop --features python

# Run the Python test suite (24 tests across 3 algorithms)
uv run python -m unittest discover -s tests -v

# Build a release wheel for distribution
uv run maturin build --release --features python
# wheel ends up in target/wheels/
```

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
use hjs_sssp::graph::{Graph, NodeId};
use hjs_sssp::sssp::{sssp, INF};

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

### Large dense graph — ~10 edges per node, weights ∈ [−100, 100]

This benchmark targets the regime where HJS's theoretical advantage begins to materialise.
Graphs are random with high density and a large weight range (many scaling phases).
BF and HJS-forced are omitted — they are orders of magnitude slower at these sizes.

| n      | m (≈10n)   | HJS      | Goldberg  |
|--------|------------|----------|-----------|
|  5 000 |  ~50 000   |  37.0 ms |  36.3 ms  |
| 12 500 | ~125 000   |   222 ms |   225 ms  |
| 25 000 | ~250 000   |   693 ms |   699 ms  |
| 62 500 | ~625 000   |  3.233 s |  3.280 s  |

HJS pulls ahead of Goldberg at n ≥ 12 500 and the gap widens slowly, consistent with the
theoretical crossover between O(log⁸ n) and O(√n) occurring well above n = 10⁵ in
practice due to constant-factor differences.

---

## Why HJS and Goldberg are so close in practice

The theoretical complexity gap between HJS (O(m log⁸ n · log(nW))) and Goldberg
(O(m √n log(nW))) is large asymptotically, but the two algorithms share nearly all of
their *actual work* at the sizes we can benchmark on a laptop:

**1. Both are potential-based bit-scaling algorithms.**
Each processes the same log₂(W) + 1 scaling phases. In every phase both algorithms build
a potential-adjusted copy of the graph and call an SSSP subroutine. The outer loop is
identical; only the inner subroutine differs.

**2. The inner subroutine is the same at small k.**
Goldberg uses `sssp_minus_one` (Bellman-Ford adapted for weights in {−1, 0, …, n})
inside each phase. HJS uses `k_sssp`, which at small k (k ≤ log⁶ n) falls back to
`sssp_few_negative` — which is also alternating Dijkstra/Bellman-Ford passes over
essentially the same graph. For graphs up to n ≈ 10⁵, log⁶ n ≈ 17⁶ ≈ 24 million,
so `k_sssp`'s recursion never fires: both algorithms run the same inner loop.

**3. The HJS recursion depth is still shallow.**
The log⁶ n threshold is enormous — larger than any n we benchmark. Even at n = 62 500,
log₂(62 500) ≈ 15.9, so log⁶ n ≈ 1.6 × 10⁷. The recursive case that distinguishes HJS
from Goldberg (building the path cover and the tiered G″ graph) is never triggered; HJS
degenerates to Goldberg for all practical input sizes on this hardware.

**4. The constant factors heavily favour Goldberg.**
Goldberg's `sssp_minus_one` is a tight, cache-friendly BF loop. HJS's `k_sssp` wrapper
adds function-call overhead, SCC decomposition, path-cover construction, and a slightly
larger working set even in the base case. These factors cancel the theoretical saving
until n is large enough for the O(√n) vs O(log⁸ n) difference to dominate them.

**In short:** HJS's advantage is exclusively in its recursion depth — but that recursion
only pays off once k (the negative-edge count per phase) exceeds log⁶ n, which requires
graphs with n ≫ 10⁶. Below that, both algorithms execute the same operations at
essentially the same cost.

---

## Python bindings (optional)

A [PyO3](https://pyo3.rs)-based Python extension is available via the `python` feature.
It accepts any object with a NetworkX-compatible `.nodes()` / `.edges(data=True)` interface.

### Build

```sh
pip install maturin
maturin develop --features python        # editable install into the active virtualenv
# or: maturin build --release --features python  # produces a wheel
```

### Usage

```python
import networkx as nx
import hjs_sssp

G = nx.DiGraph()
G.add_edge(0, 1, weight=5)
G.add_edge(1, 2, weight=-3)
G.add_edge(0, 2, weight=8)
G.add_edge(2, 3, weight=2)

# HJS 2025 — returns dict {node: distance} or None on negative cycle
dist = hjs_sssp.sssp(G, source=0)
# {0: 0, 1: 5, 2: 2, 3: 4}

# Goldberg's algorithm
dist = hjs_sssp.goldberg(G, source=0)

# Bellman-Ford
dist = hjs_sssp.bellman_ford(G, source=0)
```

All three functions share the same signature:

```python
hjs_sssp.sssp(digraph, source, *, weight="weight", default_weight=1)
```

- `digraph` — any object with `.nodes()` and `.edges(data=True)` (e.g. `nx.DiGraph`)
- `source` — source node (any hashable Python object)
- `weight` — edge attribute name to read as integer weight (default: `"weight"`)
- `default_weight` — fallback weight when the attribute is absent (default: `1`)

Returns `None` if the graph contains a negative-weight cycle; otherwise a `dict` mapping
each node to its shortest distance (`None` for unreachable nodes).

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
