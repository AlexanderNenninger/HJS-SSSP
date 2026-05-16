//! Graph projections and the layered projection construction.
//!
//! Implements the structural primitive introduced in
//! Haeupler–Jiang–Saranurak (2025), §4.1–4.2.
//!
//! # Key definitions
//!
//! A **projection** G' onto a directed graph G is a graph together with a
//! weight-preserving graph homomorphism π : V(G') → V(G): every edge
//! (u', v') ∈ E(G') satisfies (π(u'), π(v')) ∈ E(G) with the same weight.
//!
//! A projection carries **representatives**: for every original node v that
//! is *present* (i.e. π⁻¹(v) ≠ ∅) exactly one node in G' is designated
//! rep(v). Representatives are the "canonical entry points" that let us
//! stitch sub-projections together cheaply.
//!
//! The **layered projection** `Layer((G₁, …, Gₖ) → G)` (Definition 4.7)
//! combines sub-projections into one larger projection by:
//!
//! 1. Taking a disjoint union of all piece graphs.
//! 2. Assigning rep(v) to the representative of v in the *first* piece
//!    where v is present.
//! 3. Adding a cross-edge from every u' in piece i to rep(v) in piece j
//!    whenever i < j and (π(u'), v) ∈ E(G).
//!
//! Cross-edges only go *forward* through the layer ordering, so no SCC of
//! the result spans more than one layer — the clustered-diameter bound in
//! Theorem 4.5 follows directly from this fact.

use crate::graph::{Graph, NodeId};

/// A projection G' onto an original directed graph G.
pub struct Projection {
    /// The projection graph G'.
    pub graph: Graph,
    /// `proj_map\[v'.0\]` = original node π(v').
    proj_map: Vec<NodeId>,
    /// `rep\[v.0\]` = representative of original node v in G', or `None` if absent.
    rep: Vec<Option<NodeId>>,
}

impl Projection {
    // ── Accessors ─────────────────────────────────────────────────────────

    /// Number of nodes in the original graph this projects onto.
    pub fn original_size(&self) -> usize {
        self.rep.len()
    }

    /// Returns `true` if original node `v` has a representative in this projection.
    pub fn is_present(&self, v: NodeId) -> bool {
        self.rep[v.0 as usize].is_some()
    }

    /// Representative of original node `v` in this projection, or `None`.
    pub fn representative(&self, v: NodeId) -> Option<NodeId> {
        self.rep[v.0 as usize]
    }

    /// Original node that projection node `v_prime` maps to under π.
    pub fn original_of(&self, v_prime: NodeId) -> NodeId {
        self.proj_map[v_prime.0 as usize]
    }

    /// All G′-nodes that map (under π) to `orig`.
    ///
    /// With the identity projection each original node has exactly one preimage;
    /// with the full PathCover a node may appear in multiple pieces.
    pub fn preimages_of(&self, orig: NodeId) -> Vec<NodeId> {
        self.proj_map
            .iter()
            .enumerate()
            .filter(|&(_, &o)| o == orig)
            .map(|(i, _)| NodeId(i as u32))
            .collect()
    }

    // ── Constructors ──────────────────────────────────────────────────────

    /// Identity projection for the induced subgraph `G[nodes]`.
    ///
    /// Each element of `nodes` gets a unique local copy that is also its own
    /// representative. Only edges whose both endpoints are in `nodes` are
    /// included; their weights are preserved from `original`.
    ///
    /// This is the base construction used by `PathCover` for sub-problems.
    pub fn identity_subgraph(original: &Graph, nodes: &[NodeId]) -> Self {
        let original_size = original.node_count();
        let n = nodes.len();

        // Reverse map: original node id → local index, u32::MAX = absent.
        let mut local_of = vec![u32::MAX; original_size];
        for (i, &v) in nodes.iter().enumerate() {
            assert!(
                (v.0 as usize) < original_size,
                "NodeId out of range for original graph"
            );
            local_of[v.0 as usize] = i as u32;
        }

        let mut proj_graph = Graph::with_capacity(n, n);
        proj_graph.add_nodes(n);

        let proj_map: Vec<NodeId> = nodes.to_vec();

        let mut rep: Vec<Option<NodeId>> = vec![None; original_size];
        for (i, &v) in nodes.iter().enumerate() {
            rep[v.0 as usize] = Some(NodeId(i as u32));
        }

        // Include edges of the induced subgraph.
        for &v in nodes {
            let local_v = NodeId(local_of[v.0 as usize]);
            for e in original.out_edges(v) {
                let u = original.target(e);
                let local_u = local_of[u.0 as usize];
                if local_u != u32::MAX {
                    proj_graph.add_edge(local_v, NodeId(local_u), original.weight(e));
                }
            }
        }

        Projection {
            graph: proj_graph,
            proj_map,
            rep,
        }
    }

    /// Construct the **layered projection** `Layer((pieces\[0\], …, pieces\[z-1\]) → G)`.
    ///
    /// Implements Definition 4.7 of Haeupler–Jiang–Saranurak (2025).
    ///
    /// ## Construction
    ///
    /// **Step 0 — disjoint union.** All piece graphs are embedded into a
    /// single result graph. Piece i occupies result node ids
    /// `\[offsets\[i\], offsets\[i+1\])`.
    ///
    /// **Step 1 — internal edges.** Every edge inside piece i is copied into
    /// the result with source and target shifted by `offsets\[i\]`.
    ///
    /// **Step 2 — representatives.** For each original node v, its global
    /// representative is its representative in the first (lowest-index) piece
    /// where it is present.
    ///
    /// **Step 3 — cross-layer edges.** For each node u' in piece i and each
    /// outgoing edge `(π(u'), v_orig)` in the *original* graph G: if
    /// `rep(v_orig)` lives in some piece j > i, add the edge
    /// `(u'_result, rep(v_orig)_result)` with the original weight.
    ///
    /// Because cross-edges only go forward (i < j), no SCC of the result
    /// contains nodes from more than one participating layer. Inductively,
    /// if each piece is λd-clustered, so is the result.
    pub fn layer(pieces: Vec<Projection>, original: &Graph) -> Self {
        let original_size = original.node_count();
        assert!(
            pieces.iter().all(|p| p.original_size() == original_size),
            "all pieces must project onto the same original graph"
        );

        // ── Offsets ───────────────────────────────────────────────────────
        // offsets\[i\] = first result-graph node id belonging to piece i.
        let mut offsets: Vec<u32> = Vec::with_capacity(pieces.len() + 1);
        offsets.push(0);
        for p in &pieces {
            offsets.push(offsets.last().unwrap() + p.graph.node_count() as u32);
        }
        let total_nodes = *offsets.last().unwrap() as usize;

        // ── Projection map ────────────────────────────────────────────────
        // Concatenate each piece's projection map in layer order.
        let mut proj_map: Vec<NodeId> = Vec::with_capacity(total_nodes);
        for p in &pieces {
            for v_prime in p.graph.nodes() {
                proj_map.push(p.original_of(v_prime));
            }
        }

        // ── Global representatives ────────────────────────────────────────
        // For each original node, remember which result-graph node is its rep
        // and in which piece layer that rep lives.
        let mut rep_result: Vec<Option<NodeId>> = vec![None; original_size];
        let mut rep_layer: Vec<Option<usize>> = vec![None; original_size];

        for (i, p) in pieces.iter().enumerate() {
            for v_idx in 0..original_size {
                if rep_result[v_idx].is_none() {
                    if let Some(local_rep) = p.representative(NodeId(v_idx as u32)) {
                        rep_result[v_idx] = Some(NodeId(offsets[i] + local_rep.0));
                        rep_layer[v_idx] = Some(i);
                    }
                }
            }
        }

        // ── Build result graph ────────────────────────────────────────────
        let internal_edges: usize = pieces.iter().map(|p| p.graph.edge_count()).sum();
        let mut result_graph =
            Graph::with_capacity(total_nodes, internal_edges + original.edge_count());
        result_graph.add_nodes(total_nodes);

        // Step 1 — internal edges (within each piece, shifted by its offset).
        for (i, p) in pieces.iter().enumerate() {
            let offset = offsets[i];
            for e in p.graph.edges() {
                let src = NodeId(offset + p.graph.source(e).0);
                let tgt = NodeId(offset + p.graph.target(e).0);
                result_graph.add_edge(src, tgt, p.graph.weight(e));
            }
        }

        // Step 2 — cross-layer edges.
        // For u' in piece i, scan every outgoing edge of π(u') in original G.
        // If the global rep of v_orig lives in piece j > i, add a cross-edge.
        for (i, p) in pieces.iter().enumerate() {
            let offset = offsets[i];
            for u_prime in p.graph.nodes() {
                let u_orig = p.original_of(u_prime);
                let u_result = NodeId(offset + u_prime.0);
                for e in original.out_edges(u_orig) {
                    let v_orig = original.target(e);
                    if let Some(j) = rep_layer[v_orig.0 as usize] {
                        if j > i {
                            let v_result = rep_result[v_orig.0 as usize].unwrap();
                            result_graph.add_edge(u_result, v_result, original.weight(e));
                        }
                    }
                }
            }
        }

        Projection {
            graph: result_graph,
            proj_map,
            rep: rep_result,
        }
    }

    /// Replace the `graph` field with a pre-built graph that has the same
    /// node set and projection map, but different edge weights.
    ///
    /// Used by `sssp::build_path_cover` to restore original (possibly negative)
    /// weights after building a non-negative path cover.
    ///
    /// # Safety
    /// The caller must ensure `new_graph` has the same number of nodes as
    /// `self.graph`.
    pub fn with_graph(self, new_graph: Graph) -> Self {
        assert_eq!(
            new_graph.node_count(),
            self.graph.node_count(),
            "replacement graph must have the same node count"
        );
        Projection {
            graph: new_graph,
            proj_map: self.proj_map,
            rep: self.rep,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: collect all result-graph neighbours of a node.
    fn out_targets(proj: &Projection, v: NodeId) -> Vec<NodeId> {
        proj.graph
            .out_edges(v)
            .map(|e| proj.graph.target(e))
            .collect()
    }

    // ── identity_subgraph ─────────────────────────────────────────────────

    #[test]
    fn identity_full_graph() {
        // G: a →(3) b →(2) c, a →(-1) c
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 3);
        g.add_edge(a, c, -1);
        g.add_edge(b, c, 2);

        let proj = Projection::identity_subgraph(&g, &[a, b, c]);

        assert_eq!(proj.graph.node_count(), 3);
        assert_eq!(proj.graph.edge_count(), 3);

        // Every original node is present.
        assert!(proj.is_present(a));
        assert!(proj.is_present(b));
        assert!(proj.is_present(c));

        // Representatives map back to the right original nodes.
        let rep_a = proj.representative(a).unwrap();
        let rep_b = proj.representative(b).unwrap();
        let rep_c = proj.representative(c).unwrap();
        assert_eq!(proj.original_of(rep_a), a);
        assert_eq!(proj.original_of(rep_b), b);
        assert_eq!(proj.original_of(rep_c), c);
    }

    #[test]
    fn identity_proper_subgraph_excludes_cut_edges() {
        // G: a → b → c; subgraph = {a, b} should include only a→b.
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 1);
        g.add_edge(b, c, 2);

        let proj = Projection::identity_subgraph(&g, &[a, b]);

        assert_eq!(proj.graph.node_count(), 2);
        assert_eq!(proj.graph.edge_count(), 1); // only a→b

        assert!(proj.is_present(a));
        assert!(proj.is_present(b));
        assert!(!proj.is_present(c)); // c not in subgraph
    }

    // ── layer ─────────────────────────────────────────────────────────────

    /// Chain graph a→b→c split into two layers {a} and {b,c}.
    /// Expected result structure:
    ///   result nodes: 0=a', 1=b', 2=c'  (piece 0 then piece 1)
    ///   internal edges: b'→c'  (from piece 1)
    ///   cross-edge: a'→b'      (a→b in original, rep(b) is in layer 1 > 0)
    /// No cross-edge a'→c' because c is not a direct neighbour of a.
    #[test]
    fn layer_two_pieces_chain() {
        let mut g = Graph::new();
        let a = g.add_node(); // 0
        let b = g.add_node(); // 1
        let c = g.add_node(); // 2
        g.add_edge(a, b, 5);
        g.add_edge(b, c, 7);

        let piece0 = Projection::identity_subgraph(&g, &[a]);
        let piece1 = Projection::identity_subgraph(&g, &[b, c]);
        let result = Projection::layer(vec![piece0, piece1], &g);

        // 1 node in piece 0 + 2 nodes in piece 1
        assert_eq!(result.graph.node_count(), 3);

        // Representatives are in the correct pieces.
        let rep_a = result.representative(a).unwrap();
        let rep_b = result.representative(b).unwrap();
        let rep_c = result.representative(c).unwrap();

        // a' is in piece 0 → offset 0; b', c' are in piece 1 → offset 1, 2.
        assert_eq!(rep_a.0, 0);
        assert_eq!(rep_b.0, 1);
        assert_eq!(rep_c.0, 2);

        // Original nodes are preserved by proj_map.
        assert_eq!(result.original_of(rep_a), a);
        assert_eq!(result.original_of(rep_b), b);
        assert_eq!(result.original_of(rep_c), c);

        // Edge counts: 1 internal (b→c) + 1 cross (a→b).
        assert_eq!(result.graph.edge_count(), 2);

        // Cross-edge a'→b' with weight 5.
        let a_nbrs = out_targets(&result, rep_a);
        assert_eq!(a_nbrs, vec![rep_b]);
        let cross_e = result.graph.out_edges(rep_a).next().unwrap();
        assert_eq!(result.graph.weight(cross_e), 5);

        // Internal edge b'→c' with weight 7.
        let b_nbrs = out_targets(&result, rep_b);
        assert_eq!(b_nbrs, vec![rep_c]);
        let int_e = result.graph.out_edges(rep_b).next().unwrap();
        assert_eq!(result.graph.weight(int_e), 7);
    }

    /// A node present in multiple pieces should be represented by the
    /// *first* piece only; no cross-edge should be added for the later copies.
    #[test]
    fn layer_first_piece_wins_representative() {
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 1);

        // Both pieces cover {a, b}.
        let piece0 = Projection::identity_subgraph(&g, &[a, b]);
        let piece1 = Projection::identity_subgraph(&g, &[a, b]);
        let result = Projection::layer(vec![piece0, piece1], &g);

        // 2 + 2 = 4 nodes in result.
        assert_eq!(result.graph.node_count(), 4);

        // Representatives must be in piece 0 (offset 0 and 1).
        let rep_a = result.representative(a).unwrap();
        let rep_b = result.representative(b).unwrap();
        assert!(rep_a.0 < 2, "rep(a) should be in piece 0");
        assert!(rep_b.0 < 2, "rep(b) should be in piece 0");
    }

    /// No cross-edges should be added if the target's representative is in
    /// the same or an earlier layer.
    #[test]
    fn layer_no_backward_cross_edges() {
        // G: a → b. Both covered by piece 0; piece 1 only covers a.
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        g.add_edge(a, b, 3);

        let piece0 = Projection::identity_subgraph(&g, &[a, b]); // rep(a)=0, rep(b)=1
        let piece1 = Projection::identity_subgraph(&g, &[a]); // rep(a) already in piece 0

        let result = Projection::layer(vec![piece0, piece1], &g);

        // piece 0: 2 nodes (a, b); piece 1: 1 node (a copy)
        assert_eq!(result.graph.node_count(), 3);

        // rep(b) is in piece 0 (offset 1), rep(a) is in piece 0 (offset 0).
        // The a' in piece 1 (offset 2) has original edge a→b, but rep(b) is
        // in piece 0 which is *earlier* (j=0 < i=1), so NO cross-edge.
        let a_prime_in_piece1 = NodeId(2); // the copy of a in piece 1
        let a_prime_nbrs = out_targets(&result, a_prime_in_piece1);
        assert!(
            a_prime_nbrs.is_empty(),
            "no forward cross-edge should exist from piece 1's a-copy"
        );

        // The internal edge a→b in piece 0 still exists.
        assert_eq!(result.graph.edge_count(), 1);
    }

    /// Three-layer construction: piece 0 = {a}, piece 1 = {b}, piece 2 = {c}.
    /// Original: a→b→c. Cross-edges: a'→b', b'→c'.
    #[test]
    fn layer_three_pieces() {
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 1);
        g.add_edge(b, c, 2);

        let p0 = Projection::identity_subgraph(&g, &[a]);
        let p1 = Projection::identity_subgraph(&g, &[b]);
        let p2 = Projection::identity_subgraph(&g, &[c]);
        let result = Projection::layer(vec![p0, p1, p2], &g);

        assert_eq!(result.graph.node_count(), 3);
        // 2 cross-edges, no internal edges (each piece has 1 node, no intra-piece edges).
        assert_eq!(result.graph.edge_count(), 2);

        let rep_a = result.representative(a).unwrap();
        let rep_b = result.representative(b).unwrap();
        let rep_c = result.representative(c).unwrap();

        assert_eq!(out_targets(&result, rep_a), vec![rep_b]);
        assert_eq!(out_targets(&result, rep_b), vec![rep_c]);
        assert!(out_targets(&result, rep_c).is_empty());
    }

    /// Path-covering property: the lift of path a→b→c exists in the result.
    #[test]
    fn layer_covers_path() {
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 10);
        g.add_edge(b, c, 20);

        let result = Projection::layer(
            vec![
                Projection::identity_subgraph(&g, &[a]),
                Projection::identity_subgraph(&g, &[b, c]),
            ],
            &g,
        );

        let rep_a = result.representative(a).unwrap();
        let rep_b = result.representative(b).unwrap();
        let rep_c = result.representative(c).unwrap();

        // Lift of a→b→c starts at rep(a) and follows cross then internal edge.
        assert!(out_targets(&result, rep_a).contains(&rep_b));
        assert!(out_targets(&result, rep_b).contains(&rep_c));
    }
}
