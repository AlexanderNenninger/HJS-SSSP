//! PathCover(A) — the deterministic d-path cover algorithm (§4.2).
//!
//! Implements Theorem 4.5 of Haeupler–Jiang–Saranurak (2025):
//! given a directed graph G with non-negative weights, a diameter parameter d,
//! and slack λ ≥ 10000 log⁴ n, build a d-path cover G′ of G that is
//! (λ·d)-clustered and has at most (1 + 1/log n)·|E(G)| edges.
//!
//! # High-level description
//!
//! The algorithm is divide-and-conquer on the active node set A ⊆ V(G).
//!
//! **Base case** — |A| = 1: return the trivial identity subgraph [G[A]].
//!
//! **Recursive step**: pick an arbitrary node u ∈ A.  Run forward and
//! backward Dijkstra (in G[A]) to grow "balls" of increasing radius i·d,
//! stopping at the first "thin layer" where the ball does not grow by more
//! than a factor (1 + ε′).
//!
//! - **Case 1** — deg(B_out) < (1 − ε)·deg(A): recursively build path
//!   covers for *B_out* = Ball_out(u, i_out·d) and its complement
//!   *B̄_out* = A \ Ball_out(u, (i_out−1)·d) separately, then combine via
//!   `Layer((H̄_out, H_out))`.
//!
//! - **Case 2** — deg(B_out) ≥ (1 − ε)·deg(A): also run backward growing
//!   to obtain *B_in*, let M = B_out ∩ B_in, build the "middle" subgraph
//!   H_mid = G[T̃_out ∪ T̃_in] from the portions of the forward/backward
//!   SPTs that reach M, then combine via `Layer((H̃_out, H_mid, H̄_in))`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::{Graph, NodeId};
use crate::projection::Projection;
use crate::sssp::INF;

// ── Public entry point ────────────────────────────────────────────────────────

/// Build a d-path cover of `g` (Theorem 4.5).
///
/// All edge weights in `g` **must be ≥ 0**.
///
/// `d` — diameter parameter: the cover will cover every path of weight ≤ d.
/// `lambda` — slack parameter; should be ≥ 10000·log⁴ n for theoretical
/// guarantees (smaller values still produce a valid cover, just potentially
/// less clustered).
///
/// Returns a Projection G′ such that:
/// * every d-path in G is "covered" (has a lift in G′), and
/// * every SCC of G′ has diameter ≤ λ·d in G's metric.
pub fn path_cover(g: &Graph, d: i64, lambda: usize) -> Projection {
    let n = g.node_count();
    if n == 0 {
        return Projection::identity_subgraph(g, &[]);
    }
    // Precompute out-degrees in G (denominates the thin-layer check).
    let deg_g: Vec<usize> = (0..n as u32)
        .map(|v| g.out_edges(NodeId(v)).count())
        .collect();

    let active = vec![true; n];
    path_cover_rec(g, &active, &deg_g, d.max(1), lambda)
}

// ── Core recursive procedure ──────────────────────────────────────────────────

fn path_cover_rec(
    g: &Graph,
    active: &[bool],
    deg_g: &[usize],
    d: i64,
    lambda: usize,
) -> Projection {
    let n = g.node_count();
    let active_nodes: Vec<u32> = (0..n as u32).filter(|&v| active[v as usize]).collect();

    // ── Base case ─────────────────────────────────────────────────────────────
    if active_nodes.len() <= 1 {
        let ids: Vec<NodeId> = active_nodes.iter().map(|&v| NodeId(v)).collect();
        return Projection::identity_subgraph(g, &ids);
    }

    // ── Parameters (§4.2) ────────────────────────────────────────────────────
    // ε′ = 9 ln n / λ, ε = 1 / √λ
    let log_n = (n.max(2) as f64).ln().max(1.0);
    let lambda_f = (lambda.max(1)) as f64;
    let eps_prime = 9.0 * log_n / lambda_f;
    let eps = 1.0 / lambda_f.sqrt();

    // deg_G(A) = sum of out-degrees in G of all active nodes.
    let deg_a: usize = active_nodes.iter().map(|&v| deg_g[v as usize]).sum();

    // Pick an arbitrary starting node u (first in the list).
    let u_raw = active_nodes[0];
    let u = NodeId(u_raw);

    // ── Forward Dijkstra: dist_out[v] = d_{G[A]}(u, v) ───────────────────────
    let dist_out = dijkstra_subgraph(g, u, active, false);

    // Find i_out: smallest i ≥ 1 with a "thin" forward layer.
    let (i_out, deg_b_out) = find_thin_layer(&active_nodes, &dist_out, deg_g, d, eps_prime);

    // B_out  = Ball_out(u, i_out·d) in G[A]
    // B̄_out = A \ Ball_out(u, (i_out−1)·d)
    let b_out: Vec<bool> = (0..n)
        .map(|v| active[v] && dist_out[v] <= i_out as i64 * d)
        .collect();
    let radius_prev = (i_out as i64 - 1) * d; // (i_out−1)·d
    let b_bar_out: Vec<bool> = (0..n)
        .map(|v| active[v] && dist_out[v] > radius_prev)
        .collect();

    // ── Case 1 ───────────────────────────────────────────────────────────────
    // deg_G(B_out) < (1 − ε)·deg_G(A) → B_out is strictly lighter than A.
    if (deg_b_out as f64) < (1.0 - eps) * (deg_a as f64) {
        let h_out = path_cover_rec(g, &b_out, deg_g, d, lambda);
        let h_bar_out = path_cover_rec(g, &b_bar_out, deg_g, d, lambda);
        // Layer order per Definition 4.7: H̄_out first (lower index), H_out second.
        return Projection::layer(vec![h_bar_out, h_out], g);
    }

    // ── Case 2 ───────────────────────────────────────────────────────────────
    // Also grow backward until a thin layer is found.
    let dist_in = dijkstra_subgraph(g, u, active, true /* backward */);
    let (i_in, _deg_b_in) = find_thin_layer(&active_nodes, &dist_in, deg_g, d, eps_prime);

    // B_in  = Ball_in(u, i_in·d)
    // B̄_in = A \ Ball_in(u, (i_in−1)·d)
    let b_in: Vec<bool> = (0..n)
        .map(|v| active[v] && dist_in[v] <= i_in as i64 * d)
        .collect();
    let radius_in_prev = (i_in as i64 - 1) * d;
    let b_bar_in: Vec<bool> = (0..n)
        .map(|v| active[v] && dist_in[v] > radius_in_prev)
        .collect();

    // M = B_out ∩ B_in
    let m: Vec<bool> = (0..n).map(|v| b_out[v] && b_in[v]).collect();

    // Build SPTs with parent pointers, then compute T̃_out and T̃_in.
    // T_out: forward SPT from u in G[B_out].
    let (_, parent_out) = dijkstra_with_parents(g, u, &b_out, false);
    // T̃_out = ancestors of M in T_out (nodes v s.t. ∃ m∈M, v is on u→m path in T_out).
    let t_tilde_out = ancestors_in_tree(&parent_out, &m, u_raw, n);

    // T_in: backward SPT from u in G[B_in] (encodes shortest paths to u in G[B_in]).
    let (_, parent_in) = dijkstra_with_parents(g, u, &b_in, true);
    // T̃_in = ancestors of M in T_in (nodes v s.t. ∃ m∈M, v is on m→u path in T_in).
    // The backward SPT has parent pointers pointing toward u — the same
    // `ancestors_in_tree` logic applies (walk parent pointers from m to root u).
    let t_tilde_in = ancestors_in_tree(&parent_in, &m, u_raw, n);

    // H_mid = G[V(T̃_out) ∪ V(T̃_in)]: identity projection on that induced subgraph.
    let h_mid_ids: Vec<NodeId> = (0..n as u32)
        .filter(|&v| t_tilde_out[v as usize] || t_tilde_in[v as usize])
        .map(NodeId)
        .collect();
    let h_mid = Projection::identity_subgraph(g, &h_mid_ids);

    // H̄_in = PathCover(B̄_in)
    let h_bar_in = path_cover_rec(g, &b_bar_in, deg_g, d, lambda);

    // H̃_out = PathCover(B̄_out ∩ B_in)
    let h_tilde_out_active: Vec<bool> = (0..n).map(|v| b_bar_out[v] && b_in[v]).collect();
    let h_tilde_out = path_cover_rec(g, &h_tilde_out_active, deg_g, d, lambda);

    // Layer((H̃_out, H_mid, H̄_in)) per Definition 4.7.
    Projection::layer(vec![h_tilde_out, h_mid, h_bar_in], g)
}

// ── Helper: Dijkstra within G[active] ────────────────────────────────────────

/// Run Dijkstra from `source` within the subgraph G[active].
///
/// If `backward` is true, follows in-edges (computing distances TO `source`
/// in the original graph, i.e., shortest-path distances in the reverse graph).
///
/// Returns `dist[v]` = shortest distance (INF if unreachable).
fn dijkstra_subgraph(g: &Graph, source: NodeId, active: &[bool], backward: bool) -> Vec<i64> {
    let n = g.node_count();
    let mut dist = vec![INF; n];
    if !active[source.0 as usize] {
        return dist;
    }
    dist[source.0 as usize] = 0;
    let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
    heap.push(Reverse((0, source.0)));

    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        // Forward: iterate out-edges. Backward: iterate in-edges.
        if backward {
            for e in g.in_edges(NodeId(u)) {
                let v = g.source(e); // reversed direction
                if !active[v.0 as usize] {
                    continue;
                }
                let nd = d.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    heap.push(Reverse((nd, v.0)));
                }
            }
        } else {
            for e in g.out_edges(NodeId(u)) {
                let v = g.target(e);
                if !active[v.0 as usize] {
                    continue;
                }
                let nd = d.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    heap.push(Reverse((nd, v.0)));
                }
            }
        }
    }
    dist
}

/// Same as [`dijkstra_subgraph`] but also returns parent pointers.
///
/// `parent[v]` = node u such that the SPT edge is u→v (forward) or v→u (backward).
/// In both cases walking from v via parent pointers reaches the root `source`.
fn dijkstra_with_parents(
    g: &Graph,
    source: NodeId,
    active: &[bool],
    backward: bool,
) -> (Vec<i64>, Vec<Option<u32>>) {
    let n = g.node_count();
    let mut dist = vec![INF; n];
    let mut parent: Vec<Option<u32>> = vec![None; n];
    if !active[source.0 as usize] {
        return (dist, parent);
    }
    dist[source.0 as usize] = 0;
    let mut heap: BinaryHeap<Reverse<(i64, u32)>> = BinaryHeap::new();
    heap.push(Reverse((0, source.0)));

    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        if backward {
            for e in g.in_edges(NodeId(u)) {
                let v = g.source(e);
                if !active[v.0 as usize] {
                    continue;
                }
                let nd = d.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    parent[v.0 as usize] = Some(u);
                    heap.push(Reverse((nd, v.0)));
                }
            }
        } else {
            for e in g.out_edges(NodeId(u)) {
                let v = g.target(e);
                if !active[v.0 as usize] {
                    continue;
                }
                let nd = d.saturating_add(g.weight(e));
                if nd < dist[v.0 as usize] {
                    dist[v.0 as usize] = nd;
                    parent[v.0 as usize] = Some(u);
                    heap.push(Reverse((nd, v.0)));
                }
            }
        }
    }
    (dist, parent)
}

// ── Helper: thin-layer detection ─────────────────────────────────────────────

/// Find the smallest i ≥ 1 such that
/// `deg_G(Ball(i·d)) ≤ (1 + ε′) · deg_G(Ball((i−1)·d))`.
///
/// Nodes are sorted by distance once, then scanned linearly.
/// Returns `(i, deg_G(Ball(i·d)))`.
fn find_thin_layer(
    active: &[u32],
    dist: &[i64],
    deg_g: &[usize],
    d: i64,
    eps_prime: f64,
) -> (usize, usize) {
    // Sort active nodes by distance (INF last).
    let mut sorted: Vec<(i64, u32)> = active.iter().map(|&v| (dist[v as usize], v)).collect();
    sorted.sort_unstable_by_key(|&(dist, _)| dist);

    // Ball(0·d): nodes at distance ≤ 0.
    let mut prev_deg: i64 = 0;
    let mut cur_idx = 0usize;
    while cur_idx < sorted.len() && sorted[cur_idx].0 <= 0 {
        prev_deg += deg_g[sorted[cur_idx].1 as usize] as i64;
        cur_idx += 1;
    }

    let total: i64 = active.iter().map(|&v| deg_g[v as usize] as i64).sum();
    let max_iter = active.len() + 2; // safety cap — terminates in ≤ |A| steps

    for i in 1..=max_iter {
        let threshold = (i as i64).saturating_mul(d);
        let mut cur_deg = prev_deg;
        let mut next_idx = cur_idx;

        while next_idx < sorted.len() && sorted[next_idx].0 <= threshold {
            cur_deg += deg_g[sorted[next_idx].1 as usize] as i64;
            next_idx += 1;
        }

        // Thin-layer condition, or the ball has grown to cover all active nodes.
        if (cur_deg as f64) <= (1.0 + eps_prime) * (prev_deg as f64)
            || next_idx >= sorted.len()
            || cur_deg >= total
        {
            return (i, cur_deg as usize);
        }

        prev_deg = cur_deg;
        cur_idx = next_idx;
    }

    // Fallback (unreachable by Lemma 4.8 for λ large enough).
    (max_iter, total as usize)
}

// ── Helper: ancestors in SPT ─────────────────────────────────────────────────

/// Mark all ancestors of seed nodes M in a tree given by `parent` pointers.
///
/// Walking from each m ∈ M using `parent[v]` reaches the tree root `root`.
/// Marks every node visited on these walks.  Early-exit when a node is already
/// marked (to avoid repeated root traversals).
///
/// Works for both forward SPTs (T_out) and backward SPTs (T_in) because in
/// both cases `parent[v]` points one step closer to the root u.
fn ancestors_in_tree(parent: &[Option<u32>], m: &[bool], root: u32, n: usize) -> Vec<bool> {
    let mut included = vec![false; n];
    for v in 0..n as u32 {
        if !m[v as usize] {
            continue;
        }
        let mut cur = v;
        loop {
            if included[cur as usize] {
                break; // already processed this branch
            }
            included[cur as usize] = true;
            if cur == root {
                break;
            }
            match parent[cur as usize] {
                Some(p) => cur = p,
                None => break, // disconnected from root
            }
        }
    }
    included
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

    /// Verify that every path of weight ≤ d in `g` is covered by projection `p`.
    /// Checks all simple paths (for small graphs) by BFS with distance tracking.
    fn paths_covered(g: &Graph, p: &Projection, d: i64) -> bool {
        let n = g.node_count();
        // For each pair (s, t) reachable with weight ≤ d, check coverage.
        for s in 0..n as u32 {
            for t in 0..n as u32 {
                if s == t {
                    continue;
                }
                // Find all d-paths from s to t using BFS/DFS.
                // A d-path is a walk (no repeated edges but may repeat nodes)
                // of cumulative weight ≤ d.
                // We enumerate all simple paths for small n.
                let paths = enumerate_paths(g, NodeId(s), NodeId(t), d);
                for path in &paths {
                    if !path_has_lift(g, p, path) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Enumerate all simple paths from `src` to `tgt` with total weight ≤ `d`.
    fn enumerate_paths(g: &Graph, src: NodeId, tgt: NodeId, d: i64) -> Vec<Vec<NodeId>> {
        let mut result = Vec::new();
        let mut visited = vec![false; g.node_count()];
        let mut path = vec![src];
        visited[src.0 as usize] = true;
        dfs_paths(g, tgt, d, 0, &mut path, &mut visited, &mut result);
        result
    }

    fn dfs_paths(
        g: &Graph,
        tgt: NodeId,
        budget: i64,
        spent: i64,
        path: &mut Vec<NodeId>,
        visited: &mut Vec<bool>,
        result: &mut Vec<Vec<NodeId>>,
    ) {
        let cur = *path.last().unwrap();
        if cur == tgt {
            result.push(path.clone());
            return;
        }
        for e in g.out_edges(cur) {
            let v = g.target(e);
            let w = g.weight(e);
            let new_spent = spent + w;
            if new_spent <= budget && !visited[v.0 as usize] {
                visited[v.0 as usize] = true;
                path.push(v);
                dfs_paths(g, tgt, budget, new_spent, path, visited, result);
                path.pop();
                visited[v.0 as usize] = false;
            }
        }
    }

    /// Check that an original path `p_orig` (sequence of nodes) has a lift in
    /// the projection `proj`.  A lift is a sequence of G′-nodes with the same
    /// original images and each consecutive pair connected by a G′-edge.
    fn path_has_lift(_g: &Graph, proj: &Projection, p_orig: &[NodeId]) -> bool {
        if p_orig.is_empty() {
            return true;
        }
        let n_prime = proj.graph.node_count();

        // Try every G′-node that maps to p_orig[0] as a starting point.
        for start in 0..n_prime as u32 {
            if proj.original_of(NodeId(start)) != p_orig[0] {
                continue;
            }
            if lift_from(proj, p_orig, 0, NodeId(start)) {
                return true;
            }
        }
        false
    }

    fn lift_from(proj: &Projection, p: &[NodeId], idx: usize, cur_prime: NodeId) -> bool {
        if idx + 1 == p.len() {
            return true; // all nodes placed
        }
        let next_orig = p[idx + 1];
        // Find a G′ neighbour of cur_prime that maps to next_orig.
        for e in proj.graph.out_edges(cur_prime) {
            let v_prime = proj.graph.target(e);
            if proj.original_of(v_prime) == next_orig {
                if lift_from(proj, p, idx + 1, v_prime) {
                    return true;
                }
            }
        }
        false
    }

    // ── Basic tests ───────────────────────────────────────────────────────────

    #[test]
    fn cover_single_node() {
        let (g, nodes) = make_graph(1, &[]);
        let p = path_cover(&g, 5, 100);
        assert_eq!(p.graph.node_count(), 1);
        assert!(p.representative(nodes[0]).is_some());
    }

    #[test]
    fn cover_two_nodes_one_edge() {
        let (g, nodes) = make_graph(2, &[(0, 1, 3)]);
        let p = path_cover(&g, 5, 100);
        // The single edge (0→1, weight 3 ≤ 5) must be covered.
        assert!(paths_covered(&g, &p, 5), "d=5 path 0→1 not covered");
        assert!(p.representative(nodes[0]).is_some());
        assert!(p.representative(nodes[1]).is_some());
    }

    #[test]
    fn cover_chain_of_four() {
        // 0→1(1)→2(1)→3(1): all 1-paths and 2-paths must be covered.
        let (g, _) = make_graph(4, &[(0, 1, 1), (1, 2, 1), (2, 3, 1)]);
        let p = path_cover(&g, 2, 200);
        assert!(paths_covered(&g, &p, 2), "2-paths not all covered");
    }

    #[test]
    fn cover_complete_graph_k3() {
        // Fully connected 3-node graph with uniform weight 1.
        let edges = [
            (0, 1, 1),
            (0, 2, 1),
            (1, 0, 1),
            (1, 2, 1),
            (2, 0, 1),
            (2, 1, 1),
        ];
        let (g, _) = make_graph(3, &edges);
        let p = path_cover(&g, 1, 500);
        assert!(
            paths_covered(&g, &p, 1),
            "All 1-weight edges must be covered"
        );
    }

    #[test]
    fn cover_star_graph() {
        // Star: center 0, leaves 1..4, all weight 2.
        let edges: Vec<(usize, usize, i64)> =
            (1..5).flat_map(|i| vec![(0, i, 2), (i, 0, 2)]).collect();
        let (g, _) = make_graph(5, &edges);
        let p = path_cover(&g, 2, 200);
        assert!(paths_covered(&g, &p, 2), "Star edges not all covered");
    }

    #[test]
    fn all_nodes_have_representatives() {
        // Every node that is active should have a representative.
        let edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]; // cycle
        let (g, nodes) = make_graph(4, &edges);
        let p = path_cover(&g, 10, 100);
        for &nd in &nodes {
            assert!(p.representative(nd).is_some(), "node {:?} has no rep", nd);
        }
    }

    #[test]
    fn cover_disconnected_graph() {
        // Two disconnected components: {0,1} and {2,3}.
        let (g, _) = make_graph(4, &[(0, 1, 1), (2, 3, 1)]);
        let p = path_cover(&g, 1, 100);
        assert!(
            paths_covered(&g, &p, 1),
            "Disconnected: 1-paths not covered"
        );
    }

    #[test]
    fn cover_zero_weight_edges() {
        let (g, _) = make_graph(3, &[(0, 1, 0), (1, 2, 0)]);
        let p = path_cover(&g, 0, 100);
        assert!(paths_covered(&g, &p, 0), "0-weight paths not covered");
    }

    #[test]
    fn projection_maps_back_to_original() {
        let (g, nodes) = make_graph(3, &[(0, 1, 2), (1, 2, 3)]);
        let p = path_cover(&g, 5, 100);
        // Every G′-node should map back to some original node.
        for v_prime in p.graph.nodes() {
            let orig = p.original_of(v_prime);
            assert!(
                nodes.contains(&orig),
                "proj node maps to unknown original {:?}",
                orig
            );
        }
    }
}
