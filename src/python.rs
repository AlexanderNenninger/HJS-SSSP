//! Python bindings via PyO3.
//!
//! Exposes [`sssp`], [`goldberg`], and [`bellman_ford`] as Python functions
//! that accept any object with a NetworkX-compatible `.nodes()` /
//! `.edges(data=True)` interface (e.g. a `networkx.DiGraph`).
//!
//! Build with [maturin](https://github.com/PyO3/maturin):
//!
//! ```sh
//! pip install maturin
//! maturin develop --features python
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::graph::{Graph, NodeId};
use crate::sssp as algo;
use crate::sssp::INF;

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Build an internal [`Graph`] from any NetworkX-style digraph object.
///
/// Assigns each unique node a dense `u32` index in iteration order.
/// Returns `(graph, node_to_idx PyDict, idx_to_node Vec)`.
fn build_graph<'py>(
    py: Python<'py>,
    digraph: &Bound<'py, PyAny>,
    weight_attr: &str,
    default_weight: i64,
) -> PyResult<(Graph, Py<PyDict>, Vec<PyObject>)> {
    // Assign each unique node a dense integer index via a Python dict
    // so that Python's own hash/equality logic handles arbitrary node types.
    let node_to_idx = PyDict::new(py);
    let mut idx_to_node: Vec<PyObject> = Vec::new();

    for item in digraph.call_method0("nodes")?.iter()? {
        let node = item?;
        if node_to_idx.get_item(&node)?.is_none() {
            node_to_idx.set_item(&node, idx_to_node.len())?;
            idx_to_node.push(node.unbind());
        }
    }

    let n = idx_to_node.len();
    let mut g = Graph::with_capacity(n, n * 4);
    g.add_nodes(n);

    // Call digraph.edges(data=True) → iterator of (u, v, attr_dict) tuples.
    let kwargs = PyDict::new(py);
    kwargs.set_item("data", true)?;
    let edges = digraph.call_method("edges", (), Some(&kwargs))?;

    for item in edges.iter()? {
        let edge = item?;
        let u_node = edge.get_item(0usize)?;
        let v_node = edge.get_item(1usize)?;
        let attr = edge.get_item(2usize)?;

        let u_idx: u32 = node_to_idx
            .get_item(&u_node)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("edge source not in node list"))?
            .extract()?;
        let v_idx: u32 = node_to_idx
            .get_item(&v_node)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("edge target not in node list"))?
            .extract()?;

        let w: i64 = if let Ok(d) = attr.downcast::<PyDict>() {
            if let Some(val) = d.get_item(weight_attr)? {
                val.extract::<i64>()?
            } else {
                default_weight
            }
        } else {
            default_weight
        };

        g.add_edge(NodeId(u_idx), NodeId(v_idx), w);
    }

    Ok((g, node_to_idx.into(), idx_to_node))
}

/// Resolve `source` to its dense index using `node_to_idx`.
fn resolve_source(node_to_idx: &Bound<'_, PyDict>, source: &Bound<'_, PyAny>) -> PyResult<u32> {
    node_to_idx
        .get_item(source)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("source node not in graph"))?
        .extract::<u32>()
}

/// Convert a distance vector into a Python `dict` `{node: distance | None}`.
fn dist_to_dict<'py>(
    py: Python<'py>,
    dist: &[i64],
    idx_to_node: &[PyObject],
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (i, &d) in dist.iter().enumerate() {
        let node = idx_to_node[i].bind(py);
        if d < INF {
            out.set_item(node, d)?;
        } else {
            out.set_item(node, py.None())?;
        }
    }
    Ok(out)
}

// ── Public Python functions ───────────────────────────────────────────────────

/// Single-source shortest paths — HJS 2025 algorithm.
///
/// Parameters
/// ----------
/// digraph :
///     Any object with ``.nodes()`` and ``.edges(data=True)``
///     (e.g. a ``networkx.DiGraph``).
/// source :
///     Source node.
/// weight : str, optional
///     Edge attribute name for integer weights.  Default: ``"weight"``.
/// default_weight : int, optional
///     Weight to use when the attribute is absent.  Default: ``1``.
///
/// Returns
/// -------
/// dict[node, int | None] | None
///     Shortest distances from *source*, with ``None`` for unreachable nodes.
///     Returns ``None`` if the graph contains a negative-weight cycle.
#[pyfunction]
#[pyo3(signature = (digraph, source, *, weight = "weight", default_weight = 1))]
fn sssp(
    py: Python<'_>,
    digraph: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    weight: &str,
    default_weight: i64,
) -> PyResult<Option<PyObject>> {
    let (g, node_to_idx_owned, idx_to_node) = build_graph(py, digraph, weight, default_weight)?;
    let src = resolve_source(node_to_idx_owned.bind(py), source)?;
    algo::sssp(&g, NodeId(src))
        .map(|d| dist_to_dict(py, &d, &idx_to_node).map(|d| d.into()))
        .transpose()
}

/// Single-source shortest paths — Goldberg's bit-scaling algorithm.
///
/// Same interface as :func:`sssp`.  Uses Goldberg's O(m √n log(nW)) algorithm
/// instead of HJS.
#[pyfunction]
#[pyo3(signature = (digraph, source, *, weight = "weight", default_weight = 1))]
fn goldberg(
    py: Python<'_>,
    digraph: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    weight: &str,
    default_weight: i64,
) -> PyResult<Option<PyObject>> {
    let (g, node_to_idx_owned, idx_to_node) = build_graph(py, digraph, weight, default_weight)?;
    let src = resolve_source(node_to_idx_owned.bind(py), source)?;
    algo::goldberg(&g, NodeId(src))
        .map(|d| dist_to_dict(py, &d, &idx_to_node).map(|d| d.into()))
        .transpose()
}

/// Single-source shortest paths — Bellman-Ford / Moore algorithm.
///
/// Same interface as :func:`sssp`.  O(mn) — use only for small graphs or
/// when you need the simplest possible baseline.
#[pyfunction]
#[pyo3(signature = (digraph, source, *, weight = "weight", default_weight = 1))]
fn bellman_ford(
    py: Python<'_>,
    digraph: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    weight: &str,
    default_weight: i64,
) -> PyResult<Option<PyObject>> {
    let (g, node_to_idx_owned, idx_to_node) = build_graph(py, digraph, weight, default_weight)?;
    let src = resolve_source(node_to_idx_owned.bind(py), source)?;
    algo::bellman_ford(&g, NodeId(src))
        .map(|d| dist_to_dict(py, &d, &idx_to_node).map(|d| d.into()))
        .transpose()
}

// ── Module definition ─────────────────────────────────────────────────────────

#[pymodule]
fn hjs_sssp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sssp, m)?)?;
    m.add_function(wrap_pyfunction!(goldberg, m)?)?;
    m.add_function(wrap_pyfunction!(bellman_ford, m)?)?;
    Ok(())
}
