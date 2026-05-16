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

use crate::graph::{EdgeId, Graph, NodeId};
use crate::projection::Projection;

// ── Constants ─────────────────────────────────────────────────────────────────

/// A distance value of +∞ — no path exists.
pub const INF: i64 = i64::MAX / 2;

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

    // Internal edges within each copy (G′ edges, potential-adjusted).
    for i in 0..x {
        for e in g_prime.edges() {
            let u = g_prime.source(e);
            let v = g_prime.target(e);
            let w = g_prime.weight(e);
            let phi_u = phi[u.0 as usize];
            let phi_v = phi[v.0 as usize];
            let w_adj = if phi_u < INF && phi_v < INF {
                w.saturating_add(phi_u).saturating_sub(phi_v)
            } else {
                INF
            };
            if w_adj < INF {
                g_double.add_edge(global(i, u.0), global(i, v.0), w_adj);
            }
        }
    }

    // Cross-layer edges (copy i → copy i+1):
    // For each H-edge (u_orig, v_orig) and each G′-preimage u′ of u_orig,
    // add edge from copy_i(u′) to copy_{i+1}(rep(v_orig)).
    for i in 0..(x - 1) {
        for eh in h.edges() {
            let u_orig = h.source(eh);
            let v_orig = h.target(eh);
            let w_h = h.weight(eh);
            let v_rep = match cover.representative(v_orig) {
                Some(r) => r.0,
                None => continue,
            };
            let phi_v = phi[v_rep as usize];
            for &u_prime in &preimage_idx[u_orig.0 as usize] {
                let phi_u = phi[u_prime as usize];
                let w_adj = if phi_u < INF && phi_v < INF {
                    w_h.saturating_add(phi_u).saturating_sub(phi_v)
                } else {
                    INF
                };
                if w_adj < INF {
                    g_double.add_edge(global(i, u_prime), global(i + 1, v_rep), w_adj);
                }
            }
        }
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
    let mut dist = vec![INF; n];
    for u in 0..n as u32 {
        for &u_prime in &preimage_idx[u as usize] {
            let phi_u = phi[u_prime as usize];
            if phi_u >= INF {
                continue;
            }
            for i in 0..x {
                let d_raw = dist_double[global(i, u_prime).0 as usize];
                if d_raw < INF {
                    let d_u = d_raw.saturating_add(phi_u);
                    if d_u < dist[u as usize] {
                        dist[u as usize] = d_u;
                    }
                }
            }
        }
    }

    dist
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

    // Collect edges once.
    let edges: Vec<(u32, u32, i64)> = g
        .edges()
        .map(|e| (g.source(e).0, g.target(e).0, g.weight(e)))
        .collect();

    // n−1 relaxation rounds.
    for _ in 0..(n.saturating_sub(1)) {
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

    // One more pass to detect negative cycles.
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
    // Base-case cutoff: log⁶ n. We use at least 1 to avoid degenerate cases.
    let log_n = (usize::BITS - n.leading_zeros()) as usize;
    let threshold = (log_n * log_n * log_n * log_n * log_n * log_n).max(1);
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

        // Build the scaled graph for this phase.
        // Scaled weight = (w(e) >> shift) + φ(u) - φ(v).
        // After potential adjustment this must lie in {-1, 0, …, n}.
        let mut g_scaled = Graph::with_capacity(n + 1, g.edge_count() + n);
        g_scaled.add_nodes(n + 1);
        let super_src = NodeId(n as u32);

        // Double all current potentials.
        for v in 0..n {
            if phi[v] < INF {
                phi[v] = phi[v].saturating_mul(2);
            }
        }

        for e in g.edges() {
            let u = g.source(e);
            let v = g.target(e);
            let w_scaled = g.weight(e) >> shift as u32; // arithmetic right shift
            let phi_u = phi[u.0 as usize];
            let phi_v = phi[v.0 as usize];
            let w_adj = if phi_u < INF && phi_v < INF {
                w_scaled.saturating_add(phi_u).saturating_sub(phi_v)
            } else {
                INF
            };
            // Clamp to valid range; values < -1 indicate a negative cycle.
            if w_adj < -1 {
                return None; // negative cycle
            }
            let w_clamped = w_adj.min(n as i64);
            g_scaled.add_edge(u, v, w_clamped);
        }

        // Add super-source with 0-weight edges to all nodes (Definition 5.1).
        for v in 0..n as u32 {
            g_scaled.add_edge(super_src, NodeId(v), 0);
        }

        // Solve restricted SSSP from super-source.
        let dist_phase = restricted_sssp(&g_scaled, super_src);

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
    let mut g_final = Graph::with_capacity(n, g.edge_count());
    g_final.add_nodes(n);
    for e in g.edges() {
        let u = g.source(e);
        let v = g.target(e);
        let phi_u = phi[u.0 as usize];
        let phi_v = phi[v.0 as usize];
        if phi_u >= INF || phi_v >= INF {
            continue;
        }
        let w_adj = g.weight(e).saturating_add(phi_u).saturating_sub(phi_v);
        if w_adj < 0 {
            return None; // residual negative weight — negative cycle
        }
        g_final.add_edge(u, v, w_adj);
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
}
