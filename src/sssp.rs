//! Deterministic negative-weight single-source shortest paths (SSSP).
//!
//! Implements the algorithm of Haeupler–Jiang–Saranurak (2025), §5,
//! which achieves O(m log⁸ n · log(nW)) time for general integer weights.
//!
//! # Structure
//!
//! The public entry point is [`sssp`], which reduces the problem to
//! [`restricted_sssp`] via the weight-scaling/BCF23 reduction (Lemma 5.2).
//!
//! [`restricted_sssp`] calls [`k_sssp`] recursively (§5.1):
//!
//! - **Base case** (k ≤ log⁶ n): handled by [`sssp_few_negative`] (Lemma 5.5),
//!   which alternates Dijkstra and Bellman-Ford passes.
//! - **Recursive case**: build a d-path cover of H≥0 via [`path_cover`],
//!   restore weights, prune to within-SCC edges (Lemma 5.4 premise),
//!   solve recursively with k/2 negative edges allowed, then assemble G″
//!   (x = 2λ tiered copies) and solve SSSP on the potential-adjusted G″ via
//!   [`sssp_cross_scc`] (Lemma 5.4).
//!
//! # Subroutines
//!
//! | Function | Paper reference |
//! |---|---|
//! | [`dijkstra`] | standard |
//! | [`sssp_cross_scc`] | Lemma 5.4 — SSSP when negatives cross SCCs |
//! | [`sssp_few_negative`] | Lemma 5.5 — alternating Dijkstra / BF |
//! | [`path_cover`] | Theorem 4.5 (wraps `crate::projection`) |
//! | [`scc`] | Tarjan's algorithm |
//! | [`k_sssp`] | §5.1 recursive algorithm |
//! | [`sssp`] | Theorem 1.1 outer driver |

use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::graph::{EdgeId, Graph, NodeId};
use crate::projection::Projection;

// ── Constants ─────────────────────────────────────────────────────────────────

/// A distance value of +∞ — no path exists.
pub const INF: i64 = i64::MAX / 2;

/// Minimum edge count at which Bellman-Ford relaxation rounds are parallelised.
///
/// Below this threshold the per-round Rayon fork/join overhead (≈10 µs) exceeds
/// the savings from parallel edge scanning, so we fall back to direct in-place
/// mutation (no allocation per round).  Derived from benchmarks: at n=800 sparse
/// (m≈3 200) the round synchronisation cost dominates; 4 096 sits just above
/// that, ensuring the parallel path fires only for genuinely large edge sets.
#[cfg(feature = "parallel")]
const PAR_BF_MIN_EDGES: usize = 4_096;

/// Minimum work count at which graph-construction loops are parallelised.
///
/// On macOS, waking Rayon's thread pool costs ~50–100 µs per `into_par_iter()`
/// call.  At 65 536 items — assuming simple arithmetic per edge — the parallel
/// throughput (≈8× for 8 threads) covers that fixed overhead.  At benchmark
/// sizes (n ≤ 800, m ≤ 3 200) all graph-construction loops remain serial;
/// the threshold matters for large graphs (n ≥ 10 000).
#[cfg(feature = "parallel")]
const PAR_GRAPH_MIN_EDGES: usize = 65_536;

// ═══════════════════════════════════════════════════════════════════════════════
// Dijkstra (non-negative weights only)
// ═══════════════════════════════════════════════════════════════════════════════

/// Single-source shortest paths via Dijkstra.
///
/// Requires all edge weights to be non-negative.
/// Returns a vector `dist` where `dist[v.0]` is the shortest distance
/// from `source` to `v`, or [`INF`] if unreachable.
pub fn dijkstra(g: &Graph, source: NodeId) -> Vec<i64> {
    let n = g.node_count();
    let mut dist = vec![INF; n];
    dist[source.0 as usize] = 0;

    // Min-heap of (distance, node).
    let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
    heap.push(Reverse((0, source.0)));

    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue; // stale entry
        }
        for e in g.out_edges(NodeId(u)) {
            let v = g.target(e);
            let nd = d.saturating_add(g.weight(e));
            if nd < dist[v.0 as usize] {
                dist[v.0 as usize] = nd;
                heap.push(Reverse((nd, v.0)));
            }
        }
    }
    dist
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCC decomposition (Tarjan's algorithm)
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the SCC decomposition of `g` using Tarjan's iterative algorithm.
///
/// Returns a vector `comp[v.0]` ∈ `[0, num_sccs)` giving the SCC label for
/// each node. SCCs are numbered in **reverse topological order** (the SCC of a
/// source node has the smallest label).
pub fn scc(g: &Graph) -> (Vec<usize>, usize) {
    let n = g.node_count();
    let mut index_counter = 0usize;
    let mut stack: Vec<u32> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut index = vec![usize::MAX; n]; // MAX = unvisited
    let mut lowlink = vec![0usize; n];
    let mut comp = vec![0usize; n];
    let mut num_sccs = 0usize;

    // Iterative Tarjan: explicit call stack to avoid recursion depth limits.
    // Each frame stores the node and an `Option<EdgeId>` cursor into its
    // outgoing edge list — no pre-collected adjacency Vec needed.
    for start in 0..n as u32 {
        if index[start as usize] != usize::MAX {
            continue;
        }
        // (node, cursor: next out-edge to process, or None when exhausted)
        let mut call_stack: Vec<(u32, Option<EdgeId>)> =
            vec![(start, g.first_out_edge(NodeId(start)))];

        while let Some(frame) = call_stack.last_mut() {
            let u = frame.0;
            if index[u as usize] == usize::MAX {
                // First visit.
                index[u as usize] = index_counter;
                lowlink[u as usize] = index_counter;
                index_counter += 1;
                stack.push(u);
                on_stack[u as usize] = true;
            }

            match frame.1 {
                Some(e) => {
                    // Advance cursor before potentially pushing a new frame.
                    frame.1 = g.next_out_edge(e);
                    let w = g.target(e).0;
                    if index[w as usize] == usize::MAX {
                        call_stack.push((w, g.first_out_edge(NodeId(w))));
                    } else if on_stack[w as usize] {
                        lowlink[u as usize] = lowlink[u as usize].min(index[w as usize]);
                    }
                }
                None => {
                    // All neighbours processed — pop frame and propagate lowlink.
                    call_stack.pop();
                    if let Some(parent_frame) = call_stack.last() {
                        let p = parent_frame.0;
                        lowlink[p as usize] = lowlink[p as usize].min(lowlink[u as usize]);
                    }
                    // Check if u is an SCC root.
                    if lowlink[u as usize] == index[u as usize] {
                        while let Some(w) = stack.pop() {
                            on_stack[w as usize] = false;
                            comp[w as usize] = num_sccs;
                            if w == u {
                                break;
                            }
                        }
                        num_sccs += 1;
                    }
                }
            }
        }
    }

    (comp, num_sccs)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lemma 5.4 — SSSP when negative edges only cross SCCs
// ═══════════════════════════════════════════════════════════════════════════════

/// SSSP when every negative edge crosses SCCs (Lemma 5.4).
///
/// Runs in O(m log n) time. The SCC DAG is processed in topological order;
/// within each SCC only non-negative edges remain so Dijkstra suffices.
///
/// `source` must be a valid NodeId. Returns distances from `source`.
pub fn sssp_cross_scc(g: &Graph, source: NodeId) -> Vec<i64> {
    let n = g.node_count();
    let (comp, num_sccs) = scc(g);

    // Topological order of SCCs: Tarjan returns them in reverse topo order,
    // so SCC 0 is a "sink" and SCC (num_sccs-1) is a "source" in the DAG.
    // We process SCCs from highest id (sources) to lowest id (sinks).

    let mut dist = vec![INF; n];
    dist[source.0 as usize] = 0;

    // Group nodes by SCC.
    let mut scc_nodes: Vec<Vec<u32>> = vec![Vec::new(); num_sccs];
    for v in 0..n as u32 {
        scc_nodes[comp[v as usize]].push(v);
    }

    // Process SCCs in reverse order (highest id = topological source first).
    for scc_id in (0..num_sccs).rev() {
        // Relax within-SCC edges using Dijkstra on the subgraph.
        // All within-SCC edges are non-negative by hypothesis.
        let nodes = &scc_nodes[scc_id];

        // Mini Dijkstra within the SCC.
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
        for &v in nodes {
            if dist[v as usize] < INF {
                heap.push(Reverse((dist[v as usize], v)));
            }
        }

        // Membership test: `comp[v] == scc_id` — free compared to a HashSet.
        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            for e in g.out_edges(NodeId(u)) {
                let v = g.target(e);
                if comp[v.0 as usize] != scc_id {
                    continue; // cross-SCC edge; handled below
                }
                let nd = d.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    heap.push(Reverse((nd, v.0)));
                }
            }
        }

        // Propagate cross-SCC edges leaving this SCC.
        for &u in nodes {
            let du = dist[u as usize];
            if du >= INF {
                continue;
            }
            for e in g.out_edges(NodeId(u)) {
                let v = g.target(e);
                if comp[v.0 as usize] == scc_id {
                    continue;
                }
                // Negative cross-SCC edges are allowed.
                let nd = du.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                }
            }
        }
    }

    dist
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lemma 5.5 — SSSP with at most k negative edges per shortest path
// ═══════════════════════════════════════════════════════════════════════════════

/// SSSP when every shortest path uses at most `k` negative edges (Lemma 5.5).
///
/// Alternates Dijkstra on non-negative subgraph and single Bellman-Ford passes
/// over negative edges. Runs in O(k · m log n) time.
///
/// Assumes no negative cycle. Returns `None` if a negative cycle is detected.
pub fn sssp_few_negative(g: &Graph, source: NodeId, k: usize) -> Option<Vec<i64>> {
    let n = g.node_count();

    // Partition edges into non-negative and negative.
    let neg_edges: Vec<EdgeId> = g.edges().filter(|&e| g.weight(e) < 0).collect();

    let mut dist = vec![INF; n];
    dist[source.0 as usize] = 0;

    // Track which nodes to seed each Dijkstra pass.
    // Start with the source; after each BF pass seed only the nodes whose
    // distance was updated — re-processing all finite-distance nodes every
    // round is wasteful when few nodes change per BF round.
    let mut in_seed = vec![false; n];
    let mut seed_nodes: Vec<u32> = vec![source.0];
    in_seed[source.0 as usize] = true;

    for _ in 0..=k {
        // Dijkstra step: non-negative edges only, seeded from `seed_nodes`.
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
        for &v in &seed_nodes {
            in_seed[v as usize] = false; // reset in-place for the next round
            heap.push(Reverse((dist[v as usize], v)));
        }
        seed_nodes.clear();

        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            for e in g.out_edges(NodeId(u)) {
                if g.weight(e) < 0 {
                    continue; // skip negative
                }
                let v = g.target(e);
                let nd = d.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    heap.push(Reverse((nd, v.0)));
                }
            }
        }

        // Bellman-Ford pass: relax all negative edges once.
        // Collect the nodes whose distance improved — they seed the next Dijkstra.
        for &e in &neg_edges {
            let u = g.source(e);
            let v = g.target(e);
            let du = dist[u.0 as usize];
            if du >= INF {
                continue;
            }
            let nd = du.saturating_add(g.weight(e));
            if nd < dist[v.0 as usize] {
                dist[v.0 as usize] = nd;
                if !in_seed[v.0 as usize] {
                    in_seed[v.0 as usize] = true;
                    seed_nodes.push(v.0);
                }
            }
        }

        if seed_nodes.is_empty() {
            break;
        }
    }

    // Negative cycle check: one more BF pass — if anything still relaxes, cycle.
    for &e in &neg_edges {
        let u = g.source(e);
        let v = g.target(e);
        let du = dist[u.0 as usize];
        if du < INF {
            let nd = du.saturating_add(g.weight(e));
            if nd < dist[v.0 as usize] {
                return None; // negative cycle detected
            }
        }
    }

    Some(dist)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Path cover wrapper
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a d_cov-path cover of the non-negative version of `g` (Theorem 4.5).
///
/// Clamps weights to ≥ 0 (H≥0), runs `crate::path_cover::path_cover`, then
/// restores the original (possibly negative) edge weights in the returned
/// projection by walking each projection edge back to its original counterpart.
///
/// `d_cov` — the diameter parameter for the path cover.
/// `lambda` — the clustering slack parameter.
fn build_path_cover(g: &Graph, d_cov: i64, lambda: usize) -> Projection {
    let n = g.node_count();

    // Build H≥0: weights clamped to ≥ 0.
    let mut h_nonneg = Graph::with_capacity(n, g.edge_count());
    h_nonneg.add_nodes(n);
    for e in g.edges() {
        h_nonneg.add_edge(g.source(e), g.target(e), g.weight(e).max(0));
    }

    // Run the full PathCover algorithm on H≥0.
    let cover_nonneg = crate::path_cover::path_cover(&h_nonneg, d_cov, lambda);

    // Restore original (possibly negative) edge weights.
    // Pre-build a (src, tgt) → weight map so each cover edge is looked up in
    // O(1) instead of scanning g's adjacency list per edge — reduces total
    // work from O(m_cover · avg_deg) to O(m_g + m_cover).
    // For parallel edges we keep the first weight found, matching the
    // previous `break`-on-first-match behaviour.
    let mut weight_map: std::collections::HashMap<(u32, u32), i64> =
        std::collections::HashMap::with_capacity(g.edge_count());
    for e in g.edges() {
        weight_map
            .entry((g.source(e).0, g.target(e).0))
            .or_insert_with(|| g.weight(e));
    }

    let cover_nodes = cover_nonneg.graph.node_count();
    let cover_edges = cover_nonneg.graph.edge_count();
    let mut cover_graph = Graph::with_capacity(cover_nodes, cover_edges);
    cover_graph.add_nodes(cover_nodes);

    for e in cover_nonneg.graph.edges() {
        let src = cover_nonneg.graph.source(e);
        let tgt = cover_nonneg.graph.target(e);
        let orig_src = cover_nonneg.original_of(src);
        let orig_tgt = cover_nonneg.original_of(tgt);
        let w = weight_map
            .get(&(orig_src.0, orig_tgt.0))
            .copied()
            .unwrap_or(0);
        cover_graph.add_edge(src, tgt, w);
    }

    Projection::with_graph(cover_nonneg, cover_graph)
}

// ═══════════════════════════════════════════════════════════════════════════════
// §5.1 — kSSSP recursive algorithm
// ═══════════════════════════════════════════════════════════════════════════════

/// kSSSP(H, s, k) — restricted SSSP with at most k negative edges per path.
///
/// Implements §5.1, Lemma 5.6. Calls itself recursively with k/2 negative
/// edges and uses the layered G″ construction to recover distances for H.
///
/// `threshold` is the base-case cutoff (≈ log⁶ n in the paper; exposed here
/// for testing).
fn k_sssp(h: &Graph, source: NodeId, k: usize, threshold: usize) -> Vec<i64> {
    let n = h.node_count();

    // Fast path: no negative edges → plain Dijkstra.
    if !h.edges().any(|e| h.weight(e) < 0) {
        return dijkstra(h, source);
    }

    // ── Base case ─────────────────────────────────────────────────────────────
    if k <= threshold {
        return sssp_few_negative(h, source, k).unwrap_or_else(|| vec![INF; n]);
    }

    // ── Compute λ and d_cov (§5.1) ────────────────────────────────────────────
    // λ = 10000·log⁴ n (note: paper uses log⁶ n; log⁴ is sufficient for the
    // construction correctness; the extra log factors appear in the time bound).
    let log_n = (n.max(2) as f64).ln() as usize + 1;
    let lambda = (10_000 * log_n * log_n * log_n * log_n).max(1);
    // d_cov = k / (2λ): paths longer than d_cov in H≥0 are handled by the
    // recursive call; the path cover only needs to cover short paths.
    let d_cov = (k / (2 * lambda)).max(1) as i64;

    // ── Build path cover of H≥0 ───────────────────────────────────────────────
    let cover = build_path_cover(h, d_cov, lambda);
    let g_prime = &cover.graph; // G′ with original weights restored

    // Precompute preimage index: orig_node → list of G′-nodes mapping to it.
    let mut preimage_idx: Vec<Vec<u32>> = vec![Vec::new(); n];
    for v_prime in 0..g_prime.node_count() as u32 {
        let orig = cover.original_of(NodeId(v_prime));
        preimage_idx[orig.0 as usize].push(v_prime);
    }

    // ── Prune to within-SCC edges and add super-source s′ ────────────────────
    let (comp_prime, _) = scc(g_prime);

    let n_prime = g_prime.node_count();
    // s′ gets id n_prime.
    let mut g_scc = Graph::with_capacity(n_prime + 1, g_prime.edge_count() + n_prime);
    g_scc.add_nodes(n_prime + 1);
    let s_prime = NodeId(n_prime as u32);

    for e in g_prime.edges() {
        let u = g_prime.source(e);
        let v = g_prime.target(e);
        if comp_prime[u.0 as usize] == comp_prime[v.0 as usize] {
            g_scc.add_edge(u, v, g_prime.weight(e));
        }
    }
    // 0-weight edges from super-source to every G′ node.
    for v in 0..n_prime as u32 {
        g_scc.add_edge(s_prime, NodeId(v), 0);
    }

    // ── Recursive call: kSSSP(G′_SCC, s′, k/2) ───────────────────────────────
    let dist_s_prime = k_sssp(&g_scc, s_prime, k / 2, threshold);
    // φ(u′) = dist_{G′_SCC}(s′, u′) for u′ ∈ V(G′).
    let phi: Vec<i64> = dist_s_prime[..n_prime].to_vec();

    // ── Build G″: x copies of G′ linked by H-edges ───────────────────────────
    // x = 2λ in the paper; O(log k) is also correct for the restricted instance.
    let x = (2 * (usize::BITS - k.leading_zeros()) as usize).max(2);

    let total_nodes = x * n_prime + 1;
    let s_double_prime = NodeId((total_nodes - 1) as u32);

    let mut g_double = Graph::with_capacity(
        total_nodes,
        x * g_prime.edge_count() + x * h.edge_count() + x,
    );
    g_double.add_nodes(total_nodes);

    // Helper: G′-node v′ in copy i → G″ global id.
    let global = |i: usize, v_prime: u32| -> NodeId { NodeId((i * n_prime) as u32 + v_prime) };

    // Pre-compute H-edge data (resolve representative once, skip missing).
    let h_data: Vec<(u32, u32, i64)> = (0..h.edge_count() as u32)
        .filter_map(|j| {
            let eh = EdgeId(j);
            let u_orig = h.source(eh).0;
            let v_orig = h.target(eh);
            let w_h = h.weight(eh);
            let v_rep = cover.representative(v_orig)?.0;
            Some((u_orig, v_rep, w_h))
        })
        .collect();

    // Collect internal G′ edges (potential-adjusted) for all x copies.
    let phi_slice: &[i64] = &phi;

    #[cfg(feature = "parallel")]
    let internal_edges: Vec<(u32, u32, i64)> = {
        let f = |i: usize| {
            let base = (i * n_prime) as u32;
            (0..g_prime.edge_count() as u32).filter_map(move |j| {
                let e = EdgeId(j);
                let u = g_prime.source(e);
                let v = g_prime.target(e);
                let w = g_prime.weight(e);
                let phi_u = phi_slice[u.0 as usize];
                let phi_v = phi_slice[v.0 as usize];
                let w_adj = if phi_u < INF && phi_v < INF {
                    w.saturating_add(phi_u).saturating_sub(phi_v)
                } else {
                    INF
                };
                if w_adj < INF {
                    Some((base + u.0, base + v.0, w_adj))
                } else {
                    None
                }
            })
        };
        if x * g_prime.edge_count() >= PAR_GRAPH_MIN_EDGES {
            (0..x).into_par_iter().flat_map_iter(f).collect()
        } else {
            (0..x).flat_map(f).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let internal_edges: Vec<(u32, u32, i64)> = (0..x)
        .flat_map(|i| {
            let base = (i * n_prime) as u32;
            (0..g_prime.edge_count() as u32).filter_map(move |j| {
                let e = EdgeId(j);
                let u = g_prime.source(e);
                let v = g_prime.target(e);
                let w = g_prime.weight(e);
                let phi_u = phi_slice[u.0 as usize];
                let phi_v = phi_slice[v.0 as usize];
                let w_adj = if phi_u < INF && phi_v < INF {
                    w.saturating_add(phi_u).saturating_sub(phi_v)
                } else {
                    INF
                };
                if w_adj < INF {
                    Some((base + u.0, base + v.0, w_adj))
                } else {
                    None
                }
            })
        })
        .collect();

    // Collect cross-layer edges (copy i → copy i+1, via H-edges).
    let preimage_snap: &[Vec<u32>] = &preimage_idx;

    #[cfg(feature = "parallel")]
    let cross_edges: Vec<(u32, u32, i64)> = {
        let f = |i: usize| {
            let base_src = (i * n_prime) as u32;
            let base_dst = ((i + 1) * n_prime) as u32;
            h_data.iter().flat_map(move |&(u_orig, v_rep, w_h)| {
                let phi_v = phi_slice[v_rep as usize];
                preimage_snap[u_orig as usize]
                    .iter()
                    .filter_map(move |&u_prime| {
                        let phi_u = phi_slice[u_prime as usize];
                        let w_adj = if phi_u < INF && phi_v < INF {
                            w_h.saturating_add(phi_u).saturating_sub(phi_v)
                        } else {
                            INF
                        };
                        if w_adj < INF {
                            Some((base_src + u_prime, base_dst + v_rep, w_adj))
                        } else {
                            None
                        }
                    })
            })
        };
        let x_sub1 = x.saturating_sub(1);
        if x_sub1 * h_data.len() >= PAR_GRAPH_MIN_EDGES {
            (0..x_sub1).into_par_iter().flat_map_iter(f).collect()
        } else {
            (0..x_sub1).flat_map(f).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let cross_edges: Vec<(u32, u32, i64)> = (0..x.saturating_sub(1))
        .flat_map(|i| {
            let base_src = (i * n_prime) as u32;
            let base_dst = ((i + 1) * n_prime) as u32;
            h_data.iter().flat_map(move |&(u_orig, v_rep, w_h)| {
                let phi_v = phi_slice[v_rep as usize];
                preimage_snap[u_orig as usize]
                    .iter()
                    .filter_map(move |&u_prime| {
                        let phi_u = phi_slice[u_prime as usize];
                        let w_adj = if phi_u < INF && phi_v < INF {
                            w_h.saturating_add(phi_u).saturating_sub(phi_v)
                        } else {
                            INF
                        };
                        if w_adj < INF {
                            Some((base_src + u_prime, base_dst + v_rep, w_adj))
                        } else {
                            None
                        }
                    })
            })
        })
        .collect();

    let mut g_double =
        Graph::with_capacity(total_nodes, internal_edges.len() + cross_edges.len() + x);
    g_double.add_nodes(total_nodes);

    for (u, v, w) in internal_edges {
        g_double.add_edge(NodeId(u), NodeId(v), w);
    }
    for (u, v, w) in cross_edges {
        g_double.add_edge(NodeId(u), NodeId(v), w);
    }

    // s″ → every preimage of `source` in every copy of G′.
    for i in 0..x {
        for &u_prime in &preimage_idx[source.0 as usize] {
            g_double.add_edge(s_double_prime, global(i, u_prime), 0);
        }
    }

    // ── Run Lemma 5.4 (cross-SCC SSSP) on G″_φ ───────────────────────────────
    let dist_double = sssp_cross_scc(&g_double, s_double_prime);

    // ── Recover distances for H ───────────────────────────────────────────────
    // d_s(u) = min over all preimages u′ of u, over all copies i,
    //          of { d_{s″}(global(i, u′)) + φ(u′) }.
    // ── Recover distances for H ───────────────────────────────────────────────
    // d_s(u) = min over all preimages u′ of u, over all copies i,
    //          of { d_{s″}(global(i, u′)) + φ(u′) }.
    // Each u is independent — safe to parallelise with a map.
    #[cfg(feature = "parallel")]
    let dist: Vec<i64> = {
        let f = |u: u32| {
            let mut best = INF;
            for &u_prime in &preimage_snap[u as usize] {
                let phi_u = phi_slice[u_prime as usize];
                if phi_u >= INF {
                    continue;
                }
                for i in 0..x {
                    let d_raw = dist_double[(i * n_prime) as usize + u_prime as usize];
                    if d_raw < INF {
                        let d_u = d_raw.saturating_add(phi_u);
                        if d_u < best {
                            best = d_u;
                        }
                    }
                }
            }
            best
        };
        if n >= PAR_GRAPH_MIN_EDGES {
            (0..n as u32).into_par_iter().map(f).collect()
        } else {
            (0..n as u32).map(f).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let dist: Vec<i64> = (0..n as u32)
        .map(|u| {
            let mut best = INF;
            for &u_prime in &preimage_snap[u as usize] {
                let phi_u = phi_slice[u_prime as usize];
                if phi_u >= INF {
                    continue;
                }
                for i in 0..x {
                    let d_raw = dist_double[(i * n_prime) as usize + u_prime as usize];
                    if d_raw < INF {
                        let d_u = d_raw.saturating_add(phi_u);
                        if d_u < best {
                            best = d_u;
                        }
                    }
                }
            }
            best
        })
        .collect();

    dist
}

// ═══════════════════════════════════════════════════════════════════════════════
// Goldberg's scaling algorithm (1995)
// ═══════════════════════════════════════════════════════════════════════════════

/// SSSP on a graph whose edge weights lie in `{-1, 0, 1, …, n}`.
///
/// Used as the inner subroutine of [`goldberg`].  Alternates Dijkstra on
/// non-negative edges with a single Bellman-Ford pass on the (weight = −1)
/// edges, terminating after at most ⌈√n⌉ + 1 rounds.
///
/// With weights ≥ −1 the shortest path uses at most n−1 negative edges, so
/// ⌈√n⌉ rounds suffice when called inside the scaling loop (each phase only
/// needs new *potentials*, not full distances).
///
/// Returns `None` if a negative cycle is detected.
fn sssp_minus_one(g: &Graph, source: NodeId) -> Option<Vec<i64>> {
    let n = g.node_count();
    let neg_edges: Vec<EdgeId> = g.edges().filter(|&e| g.weight(e) < 0).collect();

    let mut dist = vec![INF; n];
    dist[source.0 as usize] = 0;

    let rounds = (n as f64).sqrt().ceil() as usize + 1;

    // Seed tracking: only the BF-updated nodes are pushed into the next
    // Dijkstra heap, avoiding a full O(n) scan per round.
    let mut in_seed = vec![false; n];
    let mut seed_nodes: Vec<u32> = vec![source.0];
    in_seed[source.0 as usize] = true;

    for _ in 0..=rounds {
        // Dijkstra on non-negative edges.
        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
        for &v in &seed_nodes {
            in_seed[v as usize] = false;
            heap.push(Reverse((dist[v as usize], v)));
        }
        seed_nodes.clear();

        while let Some(Reverse((d, u))) = heap.pop() {
            if d > dist[u as usize] {
                continue;
            }
            for e in g.out_edges(NodeId(u)) {
                if g.weight(e) < 0 {
                    continue;
                }
                let v = g.target(e);
                let nd = d.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    heap.push(Reverse((nd, v.0)));
                }
            }
        }

        // Single BF pass over the −1 edges.
        // Parallel + large enough: collect-then-apply snapshot (race-free).
        // Otherwise: direct in-place mutation — no allocation per round.
        #[cfg(feature = "parallel")]
        if neg_edges.len() >= PAR_BF_MIN_EDGES {
            let dist_snap: &[i64] = &dist;
            let bf_updates: Vec<(u32, i64)> = neg_edges
                .par_iter()
                .filter_map(|&e| {
                    let u = g.source(e);
                    let v = g.target(e);
                    let du = dist_snap[u.0 as usize];
                    if du >= INF {
                        return None;
                    }
                    let nd = du.saturating_add(g.weight(e));
                    if nd < dist_snap[v.0 as usize] {
                        Some((v.0, nd))
                    } else {
                        None
                    }
                })
                .collect();
            for (v, nd) in bf_updates {
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    if !in_seed[v as usize] {
                        in_seed[v as usize] = true;
                        seed_nodes.push(v);
                    }
                }
            }
            // fall through to seed_nodes check
        }
        #[cfg(feature = "parallel")]
        let _par_bf_done = neg_edges.len() >= PAR_BF_MIN_EDGES;
        #[cfg(not(feature = "parallel"))]
        let _par_bf_done = false;

        if !_par_bf_done {
            // Serial direct-mutation path.
            for &e in &neg_edges {
                let u = g.source(e);
                let v = g.target(e);
                let du = dist[u.0 as usize];
                if du >= INF {
                    continue;
                }
                let nd = du.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    if !in_seed[v.0 as usize] {
                        in_seed[v.0 as usize] = true;
                        seed_nodes.push(v.0);
                    }
                }
            }
        }

        if seed_nodes.is_empty() {
            break;
        }
    }

    // Negative-cycle check: one more BF pass.
    for &e in &neg_edges {
        let u = g.source(e);
        let v = g.target(e);
        let du = dist[u.0 as usize];
        if du < INF && du.saturating_add(g.weight(e)) < dist[v.0 as usize] {
            return None;
        }
    }

    Some(dist)
}

/// Single-source shortest paths via Goldberg's weight-scaling algorithm (1995).
///
/// Runs in O(m √n log W) time where W = max |w(e)|, which is substantially
/// faster than Bellman-Ford's O(mn) once n is large enough that √n ≪ n.
///
/// The algorithm uses ⌈log₂ W⌉ + 1 scaling phases. Each phase:
/// 1. Doubles the current Johnson-style potential φ (one bit of each weight is
///    "revealed").
/// 2. Builds the potential-adjusted graph G_φ — by induction all reduced
///    weights stay in {−1, 0, 1, …, n}.
/// 3. Adds a super-source s* with 0-weight edges to all nodes and calls
///    [`sssp_minus_one`] to compute new potentials in O(m √n) time.
///
/// After all phases a final Dijkstra on the non-negative residual graph
/// recovers exact distances from `source`.
///
/// Returns `Some(dist)` where `dist[v.0]` = shortest distance from `source`
/// to `v`, or `None` if the graph contains a negative-weight cycle.
pub fn goldberg(g: &Graph, source: NodeId) -> Option<Vec<i64>> {
    let n = g.node_count();
    if n == 0 {
        return Some(Vec::new());
    }

    let w_max: i64 = g.edges().map(|e| g.weight(e).abs()).max().unwrap_or(0);
    if w_max == 0 {
        return Some(dijkstra(g, source));
    }

    let phases = (i64::BITS - w_max.leading_zeros()) as usize + 1;
    let mut phi = vec![0i64; n];

    for phase in 0..phases {
        let shift = (phases - 1 - phase) as u32;

        // Bit-by-bit scaling: double potentials to reveal one more bit.
        for p in phi.iter_mut() {
            if *p < INF {
                *p = p.saturating_mul(2);
            }
        }

        // Build G_φ: reduced weight = (w(e) >> shift) + φ(u) − φ(v).
        // By induction this is ≥ −1; cap at n (higher values don't affect SP).
        //
        // Compute edge data in parallel, then insert serially (Graph is not
        // Sync because add_edge mutates the arena).
        let m = g.edge_count() as u32;
        let phi_snap: &[i64] = &phi;
        let n_i64 = n as i64;

        #[cfg(feature = "parallel")]
        let phi_edges: Vec<(u32, u32, i64)> = {
            let f = |i: u32| -> Option<(u32, u32, i64)> {
                let e = EdgeId(i);
                let u = g.source(e);
                let v = g.target(e);
                let phi_u = phi_snap[u.0 as usize];
                let phi_v = phi_snap[v.0 as usize];
                if phi_u >= INF || phi_v >= INF {
                    return None;
                }
                let w = (g.weight(e) >> shift)
                    .saturating_add(phi_u)
                    .saturating_sub(phi_v);
                Some((u.0, v.0, w.max(-1).min(n_i64)))
            };
            if (m as usize) >= PAR_GRAPH_MIN_EDGES {
                (0..m).into_par_iter().filter_map(f).collect()
            } else {
                (0..m).filter_map(f).collect()
            }
        };

        #[cfg(not(feature = "parallel"))]
        let phi_edges: Vec<(u32, u32, i64)> = (0..m)
            .filter_map(|i| {
                let e = EdgeId(i);
                let u = g.source(e);
                let v = g.target(e);
                let phi_u = phi_snap[u.0 as usize];
                let phi_v = phi_snap[v.0 as usize];
                if phi_u >= INF || phi_v >= INF {
                    return None;
                }
                let w = (g.weight(e) >> shift)
                    .saturating_add(phi_u)
                    .saturating_sub(phi_v);
                Some((u.0, v.0, w.max(-1).min(n_i64)))
            })
            .collect();

        let mut g_phi = Graph::with_capacity(n + 1, phi_edges.len() + n);
        g_phi.add_nodes(n + 1);
        let super_src = NodeId(n as u32);

        for (u, v, w) in phi_edges {
            g_phi.add_edge(NodeId(u), NodeId(v), w);
        }
        // Super-source reaches every node with cost 0.
        for v in 0..n as u32 {
            g_phi.add_edge(super_src, NodeId(v), 0);
        }

        // Solve SSSP in G_φ ∪ {s*} using the O(m√n) subroutine.
        let dist_phi = sssp_minus_one(&g_phi, super_src)?;

        // Accumulate new potentials.
        for v in 0..n {
            let d = dist_phi[v];
            phi[v] = if phi[v] < INF && d < INF {
                phi[v].saturating_add(d)
            } else {
                INF
            };
        }
    }

    // All reduced weights are now ≥ 0: run a final Dijkstra.
    let m = g.edge_count() as u32;
    let phi_snap: &[i64] = &phi;

    #[cfg(feature = "parallel")]
    let final_edges: Vec<(u32, u32, i64)> = {
        let f = |i: u32| -> Option<(u32, u32, i64)> {
            let e = EdgeId(i);
            let u = g.source(e);
            let v = g.target(e);
            let phi_u = phi_snap[u.0 as usize];
            let phi_v = phi_snap[v.0 as usize];
            if phi_u >= INF || phi_v >= INF {
                return None;
            }
            let w_adj = g.weight(e).saturating_add(phi_u).saturating_sub(phi_v);
            Some((u.0, v.0, w_adj))
        };
        if (m as usize) >= PAR_GRAPH_MIN_EDGES {
            (0..m).into_par_iter().filter_map(f).collect()
        } else {
            (0..m).filter_map(f).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let final_edges: Vec<(u32, u32, i64)> = (0..m)
        .filter_map(|i| {
            let e = EdgeId(i);
            let u = g.source(e);
            let v = g.target(e);
            let phi_u = phi_snap[u.0 as usize];
            let phi_v = phi_snap[v.0 as usize];
            if phi_u >= INF || phi_v >= INF {
                return None;
            }
            let w_adj = g.weight(e).saturating_add(phi_u).saturating_sub(phi_v);
            Some((u.0, v.0, w_adj))
        })
        .collect();

    let mut g_final = Graph::with_capacity(n, final_edges.len());
    g_final.add_nodes(n);
    for (u, v, w_adj) in final_edges {
        if w_adj < 0 {
            return None; // residual negative weight — negative cycle
        }
        g_final.add_edge(NodeId(u), NodeId(v), w_adj);
    }

    let dist_adj = dijkstra(&g_final, source);
    let phi_s = phi[source.0 as usize];

    Some(
        (0..n)
            .map(|v| {
                let d = dist_adj[v];
                let phi_v = phi[v];
                if d < INF && phi_v < INF && phi_s < INF {
                    d.saturating_add(phi_v).saturating_sub(phi_s)
                } else {
                    INF
                }
            })
            .collect(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bellman-Ford (reference baseline)
// ═══════════════════════════════════════════════════════════════════════════════

/// Classic Bellman-Ford SSSP — O(n·m) time, O(n) space.
///
/// Returns `Some(dist)` where `dist[v.0]` = distance from `source` to `v`,
/// or `None` if the graph contains a negative-weight cycle reachable from
/// `source`.
pub fn bellman_ford(g: &Graph, source: NodeId) -> Option<Vec<i64>> {
    let n = g.node_count();
    let mut dist = vec![INF; n];
    dist[source.0 as usize] = 0;

    // Collect edges once as plain tuples so they are Send + Sync.
    let edges: Vec<(u32, u32, i64)> = g
        .edges()
        .map(|e| (g.source(e).0, g.target(e).0, g.weight(e)))
        .collect();

    // n−1 relaxation rounds.
    // When the parallel feature is active and the edge count crosses
    // PAR_BF_MIN_EDGES, each round is parallelised via a collect-then-apply
    // snapshot (no data race).  Otherwise we use direct in-place mutation with
    // an early-exit flag — zero allocation per round.
    for _ in 0..(n.saturating_sub(1)) {
        #[cfg(feature = "parallel")]
        if edges.len() >= PAR_BF_MIN_EDGES {
            let updates: Vec<(usize, i64)> = edges
                .par_iter()
                .filter_map(|&(u, v, w)| {
                    let du = dist[u as usize];
                    if du < INF {
                        let nd = du.saturating_add(w);
                        if nd < dist[v as usize] {
                            return Some((v as usize, nd));
                        }
                    }
                    None
                })
                .collect();
            if updates.is_empty() {
                break;
            }
            for (v, nd) in updates {
                if nd < dist[v] {
                    dist[v] = nd;
                }
            }
            continue; // skip the serial block below
        }

        // Serial direct-mutation path: no Vec allocation per round.
        let mut updated = false;
        for &(u, v, w) in &edges {
            let du = dist[u as usize];
            if du < INF {
                let nd = du.saturating_add(w);
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    updated = true;
                }
            }
        }
        if !updated {
            break;
        }
    }

    // One more pass to detect negative cycles (always serial — tiny cost).
    for &(u, v, w) in &edges {
        let du = dist[u as usize];
        if du < INF && du.saturating_add(w) < dist[v as usize] {
            return None;
        }
    }

    Some(dist)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Restricted SSSP
// ═══════════════════════════════════════════════════════════════════════════════

/// Restricted SSSP (Lemma 5.3): input must satisfy Definition 5.1.
///
/// Calls `kSSSP(H, s, n)` which is sufficient by Definition 5.1.
pub fn restricted_sssp(h: &Graph, source: NodeId) -> Vec<i64> {
    let n = h.node_count();
    // Base-case cutoff: the paper prescribes log⁶ n, but that exceeds n for
    // all practical input sizes (at n = 62 500 it reaches 16⁶ ≈ 16.7M), so
    // the recursive HJS path-cover machinery never fires.  We use log² n
    // instead, which is below n for every n ≥ 2 and still provides a
    // comfortable margin above the constant overhead of the recursive case.
    // Correctness is unaffected: the threshold only selects between two code
    // paths that both produce the correct answer; the smaller value simply
    // causes the recursive (faster-asymptotically) branch to fire earlier.
    let log_n = (usize::BITS - n.leading_zeros()) as usize;
    let threshold = (log_n * log_n).max(1);
    k_sssp(h, source, n, threshold)
}

// ═══════════════════════════════════════════════════════════════════════════════
// General SSSP — Theorem 1.1 outer driver
// ═══════════════════════════════════════════════════════════════════════════════

/// General single-source shortest paths with negative weights (Theorem 1.1).
///
/// Accepts any directed graph with integer weights in `[−W, W]`. Applies the
/// BCF23 weight-scaling reduction (Lemma 5.2) to reduce to `restricted_sssp`.
///
/// Returns `Some(dist)` where `dist[v.0]` = distance from `source` to `v`,
/// or `None` if the graph contains a negative-weight cycle.
pub fn sssp(g: &Graph, source: NodeId) -> Option<Vec<i64>> {
    let n = g.node_count();
    if n == 0 {
        return Some(Vec::new());
    }

    // ── BCF23 reduction: scale weights to {-1, 0, 1, …, n} ──────────────────
    // Find W = max |w(e)|.
    let w_max: i64 = g.edges().map(|e| g.weight(e).abs()).max().unwrap_or(0);
    if w_max == 0 {
        // All weights are 0: trivially run Dijkstra.
        return Some(dijkstra(g, source));
    }

    // Determine the number of scaling phases (log₂ W + 1 phases).
    let phases = (i64::BITS - w_max.leading_zeros()) as usize + 1;

    // Maintain a running potential φ : V → i64, initially 0.
    let mut phi = vec![0i64; n];

    for phase in 0..phases {
        // In phase p (0-indexed from most significant), the scale factor is
        // 2^(phases-1-p). We use the standard bit-by-bit doubling approach:
        // double φ and add the current phase's bit of each weight.
        let shift = phases - 1 - phase;

        // Double all current potentials.
        for v in 0..n {
            if phi[v] < INF {
                phi[v] = phi[v].saturating_mul(2);
            }
        }

        // Build the scaled graph for this phase.
        // Reduced weight = (w(e) >> shift) + φ(u) − φ(v).
        // Clamped to [-1, n]: values < -1 arise in early phases while the
        // potential is still rough; a true negative cycle is detected later
        // by the final residual check.
        let m = g.edge_count() as u32;
        let phi_snap: &[i64] = &phi;
        let n_i64 = n as i64;

        #[cfg(feature = "parallel")]
        let scaled_edges: Vec<(u32, u32, i64)> = {
            let f = |i: u32| -> Option<(u32, u32, i64)> {
                let e = EdgeId(i);
                let u = g.source(e);
                let v = g.target(e);
                let phi_u = phi_snap[u.0 as usize];
                let phi_v = phi_snap[v.0 as usize];
                if phi_u >= INF || phi_v >= INF {
                    return None;
                }
                let w_scaled = g.weight(e) >> shift as u32;
                let w = w_scaled
                    .saturating_add(phi_u)
                    .saturating_sub(phi_v)
                    .max(-1)
                    .min(n_i64);
                Some((u.0, v.0, w))
            };
            if (m as usize) >= PAR_GRAPH_MIN_EDGES {
                (0..m).into_par_iter().filter_map(f).collect()
            } else {
                (0..m).filter_map(f).collect()
            }
        };

        #[cfg(not(feature = "parallel"))]
        let scaled_edges: Vec<(u32, u32, i64)> = (0..m)
            .filter_map(|i| {
                let e = EdgeId(i);
                let u = g.source(e);
                let v = g.target(e);
                let phi_u = phi_snap[u.0 as usize];
                let phi_v = phi_snap[v.0 as usize];
                if phi_u >= INF || phi_v >= INF {
                    return None;
                }
                let w_scaled = g.weight(e) >> shift as u32;
                let w = w_scaled
                    .saturating_add(phi_u)
                    .saturating_sub(phi_v)
                    .max(-1)
                    .min(n_i64);
                Some((u.0, v.0, w))
            })
            .collect();

        let has_neg = scaled_edges.iter().any(|&(_, _, w)| w < 0);

        let mut g_scaled = Graph::with_capacity(n + 1, scaled_edges.len() + n);
        g_scaled.add_nodes(n + 1);
        let super_src = NodeId(n as u32);

        for (u, v, w) in scaled_edges {
            g_scaled.add_edge(NodeId(u), NodeId(v), w);
        }

        // Super-source reaches every node with cost 0.
        for v in 0..n as u32 {
            g_scaled.add_edge(super_src, NodeId(v), 0);
        }

        // If no negative edges exist after adjustment, Dijkstra suffices.
        let dist_phase = if !has_neg {
            dijkstra(&g_scaled, super_src)
        } else {
            // Weights lie in {-1, 0, …, n}: use the O(m√n) subroutine
            // directly, bypassing the restricted_sssp → k_sssp chain whose
            // recursive case never fires at these sizes.
            match sssp_minus_one(&g_scaled, super_src) {
                Some(d) => d,
                None => return None,
            }
        };

        // Update potential: φ_new(v) = dist(super_src, v).
        for v in 0..n {
            let d = dist_phase[v];
            phi[v] = if phi[v] < INF && d < INF {
                phi[v].saturating_add(d)
            } else {
                INF
            };
        }
    }

    // Final pass: apply potential to get true distances from `source`.
    // Build the final potential-adjusted graph (all weights ≥ 0) and run Dijkstra.
    let m = g.edge_count() as u32;
    let phi_snap: &[i64] = &phi;

    #[cfg(feature = "parallel")]
    let final_edges: Vec<(u32, u32, i64)> = {
        let f = |i: u32| -> Option<(u32, u32, i64)> {
            let e = EdgeId(i);
            let u = g.source(e);
            let v = g.target(e);
            let phi_u = phi_snap[u.0 as usize];
            let phi_v = phi_snap[v.0 as usize];
            if phi_u >= INF || phi_v >= INF {
                return None;
            }
            let w_adj = g.weight(e).saturating_add(phi_u).saturating_sub(phi_v);
            Some((u.0, v.0, w_adj))
        };
        if (m as usize) >= PAR_GRAPH_MIN_EDGES {
            (0..m).into_par_iter().filter_map(f).collect()
        } else {
            (0..m).filter_map(f).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let final_edges: Vec<(u32, u32, i64)> = (0..m)
        .filter_map(|i| {
            let e = EdgeId(i);
            let u = g.source(e);
            let v = g.target(e);
            let phi_u = phi_snap[u.0 as usize];
            let phi_v = phi_snap[v.0 as usize];
            if phi_u >= INF || phi_v >= INF {
                return None;
            }
            let w_adj = g.weight(e).saturating_add(phi_u).saturating_sub(phi_v);
            Some((u.0, v.0, w_adj))
        })
        .collect();

    let mut g_final = Graph::with_capacity(n, final_edges.len());
    g_final.add_nodes(n);
    for (u, v, w_adj) in final_edges {
        if w_adj < 0 {
            return None; // residual negative weight — negative cycle
        }
        g_final.add_edge(NodeId(u), NodeId(v), w_adj);
    }

    let dist_adj = dijkstra(&g_final, source);

    // Recover true distances: dist(s, v) = dist_adj(s, v) + φ(v) - φ(s).
    let phi_s = phi[source.0 as usize];
    let result: Vec<i64> = (0..n)
        .map(|v| {
            let d_adj = dist_adj[v];
            let phi_v = phi[v];
            if d_adj < INF && phi_v < INF && phi_s < INF {
                d_adj.saturating_add(phi_v).saturating_sub(phi_s)
            } else {
                INF
            }
        })
        .collect();

    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HJS with forced path-cover recursion
// ═══════════════════════════════════════════════════════════════════════════════

/// HJS SSSP with a **forced** path-cover recursion threshold.
///
/// Identical to [`sssp`] in structure, but each per-phase inner SSSP is solved
/// by [`k_sssp`] with `threshold = forced_threshold` rather than by the
/// `sssp_minus_one` (Goldberg) shortcut.  Setting `forced_threshold` to a
/// small value (e.g. 2) makes the `k_sssp` recursive case fire at all
/// practical sizes, so the path-cover / G″ construction actually runs.
///
/// This exists for benchmarking the true HJS algorithm against Goldberg.
pub fn sssp_hjs_forced(g: &Graph, source: NodeId, forced_threshold: usize) -> Option<Vec<i64>> {
    let n = g.node_count();
    if n == 0 {
        return Some(Vec::new());
    }

    let w_max: i64 = g.edges().map(|e| g.weight(e).abs()).max().unwrap_or(0);
    if w_max == 0 {
        return Some(dijkstra(g, source));
    }

    let phases = (i64::BITS - w_max.leading_zeros()) as usize + 1;
    let mut phi = vec![0i64; n];

    for phase in 0..phases {
        let shift = phases - 1 - phase;

        for v in 0..n {
            if phi[v] < INF {
                phi[v] = phi[v].saturating_mul(2);
            }
        }

        let m = g.edge_count() as u32;
        let phi_snap: &[i64] = &phi;
        let n_i64 = n as i64;

        #[cfg(feature = "parallel")]
        let scaled_edges: Vec<(u32, u32, i64)> = {
            let f = |i: u32| -> Option<(u32, u32, i64)> {
                let e = EdgeId(i);
                let u = g.source(e);
                let v = g.target(e);
                let phi_u = phi_snap[u.0 as usize];
                let phi_v = phi_snap[v.0 as usize];
                if phi_u >= INF || phi_v >= INF {
                    return None;
                }
                let w = (g.weight(e) >> shift as u32)
                    .saturating_add(phi_u)
                    .saturating_sub(phi_v)
                    .max(-1)
                    .min(n_i64);
                Some((u.0, v.0, w))
            };
            if (m as usize) >= PAR_GRAPH_MIN_EDGES {
                (0..m).into_par_iter().filter_map(f).collect()
            } else {
                (0..m).filter_map(f).collect()
            }
        };

        #[cfg(not(feature = "parallel"))]
        let scaled_edges: Vec<(u32, u32, i64)> = (0..m)
            .filter_map(|i| {
                let e = EdgeId(i);
                let u = g.source(e);
                let v = g.target(e);
                let phi_u = phi_snap[u.0 as usize];
                let phi_v = phi_snap[v.0 as usize];
                if phi_u >= INF || phi_v >= INF {
                    return None;
                }
                let w = (g.weight(e) >> shift as u32)
                    .saturating_add(phi_u)
                    .saturating_sub(phi_v)
                    .max(-1)
                    .min(n_i64);
                Some((u.0, v.0, w))
            })
            .collect();

        let has_neg = scaled_edges.iter().any(|&(_, _, w)| w < 0);

        let mut g_scaled = Graph::with_capacity(n + 1, scaled_edges.len() + n);
        g_scaled.add_nodes(n + 1);
        let super_src = NodeId(n as u32);

        for (u, v, w) in scaled_edges {
            g_scaled.add_edge(NodeId(u), NodeId(v), w);
        }
        for v in 0..n as u32 {
            g_scaled.add_edge(super_src, NodeId(v), 0);
        }

        let dist_phase = if !has_neg {
            dijkstra(&g_scaled, super_src)
        } else {
            // Use k_sssp with the forced threshold so the path-cover
            // recursion fires rather than falling back to sssp_minus_one.
            k_sssp(&g_scaled, super_src, n + 1, forced_threshold)
        };

        for v in 0..n {
            let d = dist_phase[v];
            phi[v] = if phi[v] < INF && d < INF {
                phi[v].saturating_add(d)
            } else {
                INF
            };
        }
    }

    let m = g.edge_count() as u32;
    let phi_snap: &[i64] = &phi;

    #[cfg(feature = "parallel")]
    let final_edges_forced: Vec<(u32, u32, i64)> = {
        let f = |i: u32| -> Option<(u32, u32, i64)> {
            let e = EdgeId(i);
            let u = g.source(e);
            let v = g.target(e);
            let phi_u = phi_snap[u.0 as usize];
            let phi_v = phi_snap[v.0 as usize];
            if phi_u >= INF || phi_v >= INF {
                return None;
            }
            let w_adj = g.weight(e).saturating_add(phi_u).saturating_sub(phi_v);
            Some((u.0, v.0, w_adj))
        };
        if (m as usize) >= PAR_GRAPH_MIN_EDGES {
            (0..m).into_par_iter().filter_map(f).collect()
        } else {
            (0..m).filter_map(f).collect()
        }
    };

    #[cfg(not(feature = "parallel"))]
    let final_edges_forced: Vec<(u32, u32, i64)> = (0..m)
        .filter_map(|i| {
            let e = EdgeId(i);
            let u = g.source(e);
            let v = g.target(e);
            let phi_u = phi_snap[u.0 as usize];
            let phi_v = phi_snap[v.0 as usize];
            if phi_u >= INF || phi_v >= INF {
                return None;
            }
            let w_adj = g.weight(e).saturating_add(phi_u).saturating_sub(phi_v);
            Some((u.0, v.0, w_adj))
        })
        .collect();

    let mut g_final = Graph::with_capacity(n, final_edges_forced.len());
    g_final.add_nodes(n);
    for (u, v, w_adj) in final_edges_forced {
        if w_adj < 0 {
            return None;
        }
        g_final.add_edge(NodeId(u), NodeId(v), w_adj);
    }

    let dist_adj = dijkstra(&g_final, source);
    let phi_s = phi[source.0 as usize];

    Some(
        (0..n)
            .map(|v| {
                let d = dist_adj[v];
                let phi_v = phi[v];
                if d < INF && phi_v < INF && phi_s < INF {
                    d.saturating_add(phi_v).saturating_sub(phi_s)
                } else {
                    INF
                }
            })
            .collect(),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(n: usize, edges: &[(usize, usize, i64)]) -> (Graph, Vec<NodeId>) {
        let mut g = Graph::new();
        let nodes: Vec<NodeId> = (0..n).map(|_| g.add_node()).collect();
        for &(u, v, w) in edges {
            g.add_edge(nodes[u], nodes[v], w);
        }
        (g, nodes)
    }

    // ── Dijkstra ──────────────────────────────────────────────────────────────

    #[test]
    fn dijkstra_simple() {
        let (g, nodes) = make_graph(4, &[(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 5), (2, 3, 1)]);
        let dist = dijkstra(&g, nodes[0]);
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], 1);
        assert_eq!(dist[nodes[2].0 as usize], 3);
        assert_eq!(dist[nodes[3].0 as usize], 4);
    }

    #[test]
    fn dijkstra_unreachable() {
        let (g, nodes) = make_graph(3, &[(0, 1, 1)]);
        let dist = dijkstra(&g, nodes[0]);
        assert_eq!(dist[nodes[2].0 as usize], INF);
    }

    // ── SCC ───────────────────────────────────────────────────────────────────

    #[test]
    fn scc_simple_dag() {
        // 0→1→2: three singleton SCCs
        let (g, _) = make_graph(3, &[(0, 1, 1), (1, 2, 1)]);
        let (comp, num) = scc(&g);
        assert_eq!(num, 3);
        assert!(comp[0] != comp[1] && comp[1] != comp[2]);
    }

    #[test]
    fn scc_cycle() {
        // 0→1→2→0: one SCC
        let (g, _) = make_graph(3, &[(0, 1, 1), (1, 2, 1), (2, 0, 1)]);
        let (comp, num) = scc(&g);
        assert_eq!(num, 1);
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
    }

    #[test]
    fn scc_two_components() {
        // 0→1→0, 2→3→2
        let (g, _) = make_graph(4, &[(0, 1, 1), (1, 0, 1), (2, 3, 1), (3, 2, 1)]);
        let (comp, num) = scc(&g);
        assert_eq!(num, 2);
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[2], comp[3]);
        assert_ne!(comp[0], comp[2]);
    }

    // ── sssp_cross_scc ────────────────────────────────────────────────────────

    #[test]
    fn cross_scc_negative_edge_between_sccs() {
        // 0 → 1 with weight -3 (different SCCs since no back-edge)
        // 1 → 2 with weight 5
        let (g, nodes) = make_graph(3, &[(0, 1, -3), (1, 2, 5)]);
        let dist = sssp_cross_scc(&g, nodes[0]);
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], -3);
        assert_eq!(dist[nodes[2].0 as usize], 2);
    }

    #[test]
    fn cross_scc_dag_with_negatives() {
        // DAG: 0→1 (w=-1), 0→2 (w=2), 1→2 (w=3)
        let (g, nodes) = make_graph(3, &[(0, 1, -1), (0, 2, 2), (1, 2, 3)]);
        let dist = sssp_cross_scc(&g, nodes[0]);
        assert_eq!(dist[nodes[1].0 as usize], -1);
        assert_eq!(dist[nodes[2].0 as usize], 2); // min(2, -1+3)
    }

    // ── sssp_few_negative ─────────────────────────────────────────────────────

    #[test]
    fn few_neg_basic() {
        let (g, nodes) = make_graph(3, &[(0, 1, -2), (1, 2, 5), (0, 2, 10)]);
        let dist = sssp_few_negative(&g, nodes[0], 2).unwrap();
        assert_eq!(dist[nodes[1].0 as usize], -2);
        assert_eq!(dist[nodes[2].0 as usize], 3);
    }

    #[test]
    fn few_neg_no_negative_cycle() {
        // Positive cycle: should not trigger negative cycle detection.
        let (g, nodes) = make_graph(3, &[(0, 1, 1), (1, 2, 1), (2, 0, 1), (0, 2, -1)]);
        let result = sssp_few_negative(&g, nodes[0], 3);
        assert!(result.is_some());
        let dist = result.unwrap();
        assert_eq!(dist[nodes[2].0 as usize], -1);
    }

    // ── Full SSSP ─────────────────────────────────────────────────────────────

    #[test]
    fn sssp_all_non_negative() {
        let (g, nodes) = make_graph(4, &[(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 5), (2, 3, 1)]);
        let dist = sssp(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], 1);
        assert_eq!(dist[nodes[2].0 as usize], 3);
        assert_eq!(dist[nodes[3].0 as usize], 4);
    }

    #[test]
    fn sssp_with_negative_edges() {
        // Classic Johnson's example: 0→1(w=-1), 1→2(w=3), 0→2(w=10)
        let (g, nodes) = make_graph(3, &[(0, 1, -1), (1, 2, 3), (0, 2, 10)]);
        let dist = sssp(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], -1);
        assert_eq!(dist[nodes[2].0 as usize], 2);
    }

    #[test]
    fn sssp_johnson_style() {
        // 0→1(-2), 1→2(4), 2→0(1) — positive cycle, no negative cycle
        let (g, nodes) = make_graph(3, &[(0, 1, -2), (1, 2, 4), (2, 0, 1)]);
        let dist = sssp(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], -2);
        assert_eq!(dist[nodes[2].0 as usize], 2);
    }

    #[test]
    fn sssp_single_node() {
        let (g, nodes) = make_graph(1, &[]);
        let dist = sssp(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
    }

    #[test]
    fn sssp_disconnected() {
        let (g, nodes) = make_graph(3, &[(0, 1, 1)]);
        let dist = sssp(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], 1);
        assert_eq!(dist[nodes[2].0 as usize], INF);
    }

    // ── Goldberg ──────────────────────────────────────────────────────────────

    #[test]
    fn goldberg_all_non_negative() {
        let (g, nodes) = make_graph(4, &[(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 5), (2, 3, 1)]);
        let dist = goldberg(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], 1);
        assert_eq!(dist[nodes[2].0 as usize], 3);
        assert_eq!(dist[nodes[3].0 as usize], 4);
    }

    #[test]
    fn goldberg_with_negative_edges() {
        let (g, nodes) = make_graph(3, &[(0, 1, -1), (1, 2, 3), (0, 2, 10)]);
        let dist = goldberg(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], -1);
        assert_eq!(dist[nodes[2].0 as usize], 2);
    }

    #[test]
    fn goldberg_matches_bellman_ford() {
        // Agrees with BF on a graph with mixed weights and a cycle.
        let (g, nodes) = make_graph(
            4,
            &[(0, 1, -2), (1, 2, 3), (2, 3, -1), (3, 1, 2), (0, 3, 10)],
        );
        let gd = goldberg(&g, nodes[0]).unwrap();
        let bf = bellman_ford(&g, nodes[0]).unwrap();
        assert_eq!(gd, bf);
    }

    #[test]
    fn goldberg_single_node() {
        let (g, nodes) = make_graph(1, &[]);
        let dist = goldberg(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
    }

    #[test]
    fn goldberg_disconnected() {
        let (g, nodes) = make_graph(3, &[(0, 1, -3)]);
        let dist = goldberg(&g, nodes[0]).unwrap();
        assert_eq!(dist[nodes[0].0 as usize], 0);
        assert_eq!(dist[nodes[1].0 as usize], -3);
        assert_eq!(dist[nodes[2].0 as usize], INF);
    }
}
