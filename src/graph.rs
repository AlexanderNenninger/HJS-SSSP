/// Index into the node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Index into the edge arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

/// A node stored in the arena.
struct Node {
    /// Head of the intrusive singly-linked list of outgoing edges.
    first_out: Option<EdgeId>,
    /// Head of the intrusive singly-linked list of incoming edges.
    first_in: Option<EdgeId>,
}

/// An edge stored in the arena.
struct Edge {
    source: NodeId,
    target: NodeId,
    weight: i64,
    /// Next outgoing edge from the same source (linked list through the arena).
    next_out: Option<EdgeId>,
    /// Next incoming edge into the same target (linked list through the arena).
    next_in: Option<EdgeId>,
}

/// Directed weighted graph backed by two flat arenas (nodes and edges).
///
/// No heap allocation is performed per node or per edge beyond the arena `Vec`s
/// themselves. Adjacency lists are intrusive linked lists threaded through the
/// edge arena.
pub struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Create an empty graph with pre-allocated capacity.
    pub fn with_capacity(node_cap: usize, edge_cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(node_cap),
            edges: Vec::with_capacity(edge_cap),
        }
    }

    /// Add a new isolated node and return its id.
    pub fn add_node(&mut self) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            first_out: None,
            first_in: None,
        });
        id
    }

    /// Add `count` isolated nodes, returning the id of the first one.
    /// The returned ids are `first`, `first+1`, …, `first+count-1`.
    pub fn add_nodes(&mut self, count: usize) -> NodeId {
        let first = NodeId(self.nodes.len() as u32);
        self.nodes.reserve(count);
        for _ in 0..count {
            self.nodes.push(Node {
                first_out: None,
                first_in: None,
            });
        }
        first
    }

    /// Add a directed edge from `source` to `target` with the given `weight`.
    /// Weights may be negative. Panics if either node id is out of range.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, weight: i64) -> EdgeId {
        assert!(
            (source.0 as usize) < self.nodes.len(),
            "source NodeId out of range"
        );
        assert!(
            (target.0 as usize) < self.nodes.len(),
            "target NodeId out of range"
        );

        let id = EdgeId(self.edges.len() as u32);

        let prev_out = self.nodes[source.0 as usize].first_out;
        let prev_in = self.nodes[target.0 as usize].first_in;

        self.edges.push(Edge {
            source,
            target,
            weight,
            next_out: prev_out,
            next_in: prev_in,
        });

        self.nodes[source.0 as usize].first_out = Some(id);
        self.nodes[target.0 as usize].first_in = Some(id);

        id
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Source node of an edge.
    pub fn source(&self, e: EdgeId) -> NodeId {
        self.edges[e.0 as usize].source
    }

    /// Target node of an edge.
    pub fn target(&self, e: EdgeId) -> NodeId {
        self.edges[e.0 as usize].target
    }

    /// Weight of an edge.
    pub fn weight(&self, e: EdgeId) -> i64 {
        self.edges[e.0 as usize].weight
    }

    /// Iterate over the outgoing edges of `node`.
    pub fn out_edges(&self, node: NodeId) -> EdgeIter<'_> {
        EdgeIter {
            graph: self,
            current: self.nodes[node.0 as usize].first_out,
            direction: Direction::Out,
        }
    }

    /// Iterate over the incoming edges of `node`.
    pub fn in_edges(&self, node: NodeId) -> EdgeIter<'_> {
        EdgeIter {
            graph: self,
            current: self.nodes[node.0 as usize].first_in,
            direction: Direction::In,
        }
    }

    /// Iterate over all node ids.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> {
        (0..self.nodes.len() as u32).map(NodeId)
    }

    /// Iterate over all edge ids.
    pub fn edges(&self) -> impl Iterator<Item = EdgeId> {
        (0..self.edges.len() as u32).map(EdgeId)
    }

    /// First outgoing edge from `node`, or `None` if there are none.
    ///
    /// Combined with \[`next_out_edge`\](Self::next_out_edge) this allows
    /// iterating out-edges without a heap-allocated adjacency list.
    pub fn first_out_edge(&self, node: NodeId) -> Option<EdgeId> {
        self.nodes[node.0 as usize].first_out
    }

    /// The outgoing edge that follows `e` from the same source node, or `None`
    /// if `e` is the last outgoing edge of that node.
    pub fn next_out_edge(&self, e: EdgeId) -> Option<EdgeId> {
        self.edges[e.0 as usize].next_out
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Iterator ─────────────────────────────────────────────────────────────────

enum Direction {
    Out,
    In,
}

/// Iterator over the outgoing or incoming edges of a node.
pub struct EdgeIter<'g> {
    graph: &'g Graph,
    current: Option<EdgeId>,
    direction: Direction,
}

impl<'g> Iterator for EdgeIter<'g> {
    type Item = EdgeId;

    fn next(&mut self) -> Option<EdgeId> {
        let id = self.current?;
        let edge = &self.graph.edges[id.0 as usize];
        self.current = match self.direction {
            Direction::Out => edge.next_out,
            Direction::In => edge.next_in,
        };
        Some(id)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_nodes_and_edges() {
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();

        let e1 = g.add_edge(a, b, 3);
        let e2 = g.add_edge(a, c, -1);
        let e3 = g.add_edge(b, c, 2);

        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 3);

        assert_eq!(g.weight(e1), 3);
        assert_eq!(g.weight(e2), -1);
        assert_eq!(g.weight(e3), 2);
    }

    #[test]
    fn out_edges_correct() {
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, b, 1);
        g.add_edge(a, c, 2);
        g.add_edge(b, c, 3);

        let out_a: Vec<_> = g.out_edges(a).collect();
        assert_eq!(out_a.len(), 2);
        // Both outgoing edges from a reach b or c
        for e in out_a {
            assert!(g.target(e) == b || g.target(e) == c);
        }

        let out_b: Vec<_> = g.out_edges(b).collect();
        assert_eq!(out_b.len(), 1);
        assert_eq!(g.target(out_b[0]), c);

        let out_c: Vec<_> = g.out_edges(c).collect();
        assert_eq!(out_c.len(), 0);
    }

    #[test]
    fn in_edges_correct() {
        let mut g = Graph::new();
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_edge(a, c, 10);
        g.add_edge(b, c, 20);

        let in_c: Vec<_> = g.in_edges(c).collect();
        assert_eq!(in_c.len(), 2);

        let in_a: Vec<_> = g.in_edges(a).collect();
        assert_eq!(in_a.len(), 0);
    }

    #[test]
    fn add_nodes_bulk() {
        let mut g = Graph::new();
        let first = g.add_nodes(5);
        assert_eq!(first.0, 0);
        assert_eq!(g.node_count(), 5);
    }

    #[test]
    #[should_panic]
    fn out_of_range_edge_panics() {
        let mut g = Graph::new();
        let a = g.add_node();
        g.add_edge(a, NodeId(99), 0);
    }
}
