//! Benchmark graph generators.
//!
//! Each generator follows the structure of the corresponding
//! [DIMACS 9th-challenge SPLIB][splib] generator, differing only in the choice
//! of PRNG (xorshift64 here vs. the Park–Miller multiplicative congruential
//! generator in the original C programs).
//!
//! [splib]: http://www.diag.uniroma1.it/challenge9/generators.shtml

use crate::graph::{Graph, NodeId};

// ── RNG ───────────────────────────────────────────────────────────────────────

/// xorshift64 step: period 2⁶⁴ − 1, adequate for graph generation.
#[inline]
fn next(r: &mut u64) -> u64 {
    *r ^= *r << 13;
    *r ^= *r >> 7;
    *r ^= *r << 17;
    *r
}

// ── Generators ────────────────────────────────────────────────────────────────

/// Random directed graph with `n` nodes and up to `m` arcs (**sprand**).
///
/// Arc endpoints `(u, v)` are drawn independently and uniformly from
/// `[0, n)`.  Self-loops are discarded, so the actual arc count is slightly
/// below `m` (by roughly `m / n` on average).  Weights are drawn uniformly
/// from `[-w_max, w_max]`.
///
/// This matches the structure of Goldberg's **sprand** generator from SPLIB:
/// both draw `m` uniform random pairs and skip self-loops.
///
/// # Examples
///
/// ```
/// use hjs_sssp::generators::make_random_graph;
///
/// let n = 50;
/// let m = 4 * n;
/// let w_max = 10;
/// let (g, src) = make_random_graph(n, m, w_max, 42);
///
/// assert_eq!(g.node_count(), n);
/// assert!(g.edge_count() <= m);
/// assert!(g.edges().all(|e| g.source(e) != g.target(e)));
/// assert!(g.edges().all(|e| g.weight(e).abs() <= w_max as i64));
/// ```
pub fn make_random_graph(n: usize, m: usize, w_max: i64, seed: u64) -> (Graph, NodeId) {
    let mut g = Graph::with_capacity(n, m);
    g.add_nodes(n);
    let mut rng = seed;
    for _ in 0..m {
        let u = (next(&mut rng) as usize % n) as u32;
        let v = (next(&mut rng) as usize % n) as u32;
        if u == v {
            continue;
        }
        let w = (next(&mut rng) % (2 * w_max as u64 + 1)) as i64 - w_max;
        g.add_edge(NodeId(u), NodeId(v), w);
    }
    (g, NodeId(0))
}

/// Path graph `0 → 1 → 2 → … → (n-1)` with negative-capable weights
/// (**sppath**).
///
/// Weights are drawn uniformly from `[-w_max, w_max]`.  The graph is a DAG
/// and therefore contains no cycles of any kind.
///
/// # Examples
///
/// ```
/// use hjs_sssp::generators::make_path_graph;
///
/// let n = 20;
/// let w_max = 5;
/// let (g, src) = make_path_graph(n, w_max, 99);
///
/// assert_eq!(g.node_count(), n);
/// assert_eq!(g.edge_count(), n - 1);
/// assert!(g.edges().all(|e| g.weight(e).abs() <= w_max as i64));
/// // Each edge i connects node i to node i+1.
/// assert!(g.edges().all(|e| {
///     g.target(e).0 == g.source(e).0 + 1
/// }));
/// ```
pub fn make_path_graph(n: usize, w_max: i64, seed: u64) -> (Graph, NodeId) {
    let mut g = Graph::with_capacity(n, n - 1);
    g.add_nodes(n);
    let mut rng = seed;
    for u in 0..(n - 1) as u32 {
        let w = (next(&mut rng) % (2 * w_max as u64 + 1)) as i64 - w_max;
        g.add_edge(NodeId(u), NodeId(u + 1), w);
    }
    (g, NodeId(0))
}

/// Directed `rows × cols` grid graph with negative-capable weights
/// (**spgrid**).
///
/// Each node `(r, c)` has an edge to `(r, c+1)` (right) and to `(r+1, c)`
/// (down), with weights drawn uniformly from `[-w_max, w_max]`.  All edges
/// increase the row-major index, so the graph is a DAG with no negative
/// cycles.
///
/// The total arc count is `rows*(cols-1) + (rows-1)*cols`.
///
/// # Examples
///
/// ```
/// use hjs_sssp::generators::make_grid_graph;
///
/// let (rows, cols) = (5, 6);
/// let w_max = 3;
/// let (g, src) = make_grid_graph(rows, cols, w_max, 7);
///
/// assert_eq!(g.node_count(), rows * cols);
/// assert_eq!(g.edge_count(), rows * (cols - 1) + (rows - 1) * cols);
/// assert!(g.edges().all(|e| g.weight(e).abs() <= w_max as i64));
/// ```
pub fn make_grid_graph(rows: usize, cols: usize, w_max: i64, seed: u64) -> (Graph, NodeId) {
    let n = rows * cols;
    let mut g = Graph::with_capacity(n, 2 * n);
    g.add_nodes(n);
    let idx = |r: usize, c: usize| NodeId((r * cols + c) as u32);
    let mut rng = seed;
    for r in 0..rows {
        for c in 0..cols {
            let mut w = || (next(&mut rng) % (2 * w_max as u64 + 1)) as i64 - w_max;
            if c + 1 < cols {
                g.add_edge(idx(r, c), idx(r, c + 1), w());
            }
            if r + 1 < rows {
                g.add_edge(idx(r, c), idx(r + 1, c), w());
            }
        }
    }
    (g, NodeId(0))
}
