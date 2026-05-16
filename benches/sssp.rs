use std::time::Duration;

use criterion::{BenchmarkId, Criterion, SamplingMode, black_box, criterion_group, criterion_main};
use hjs_path_finding::graph::{Graph, NodeId};
use hjs_path_finding::sssp::{bellman_ford, goldberg, sssp, sssp_hjs_forced};

// ── Graph generators ──────────────────────────────────────────────────────────

/// Dense random-ish graph on `n` nodes and ~4n directed edges.
/// Weights in [-w_max, w_max], seeded deterministically.
fn make_sparse_graph(n: usize, w_max: i64, seed: u64) -> (Graph, NodeId) {
    let mut g = Graph::with_capacity(n, 4 * n);
    g.add_nodes(n);
    let mut rng = seed;
    let next = |r: &mut u64| -> u64 {
        // xorshift64
        *r ^= *r << 13;
        *r ^= *r >> 7;
        *r ^= *r << 17;
        *r
    };
    // Each node gets ~4 random out-edges.
    for u in 0..n as u32 {
        for _ in 0..4 {
            let v = (next(&mut rng) as usize % n) as u32;
            let w = (next(&mut rng) % (2 * w_max as u64 + 1)) as i64 - w_max;
            if u != v {
                g.add_edge(NodeId(u), NodeId(v), w);
            }
        }
    }
    (g, NodeId(0))
}

/// Path graph 0→1→2→…→(n-1) with weights in [-w_max, w_max].
/// Guaranteed no negative cycle (strictly increasing path).
fn make_path_graph(n: usize, w_max: i64, seed: u64) -> (Graph, NodeId) {
    let mut g = Graph::with_capacity(n, n - 1);
    g.add_nodes(n);
    let mut rng = seed;
    let next = |r: &mut u64| -> u64 {
        *r ^= *r << 13;
        *r ^= *r >> 7;
        *r ^= *r << 17;
        *r
    };
    for u in 0..(n - 1) as u32 {
        let w = (next(&mut rng) % (2 * w_max as u64 + 1)) as i64 - w_max;
        g.add_edge(NodeId(u), NodeId(u + 1), w);
    }
    (g, NodeId(0))
}

/// Grid graph: nodes (row, col) in an r×c grid, edges right and down with
/// small random weights.
fn make_grid_graph(rows: usize, cols: usize, w_max: i64, seed: u64) -> (Graph, NodeId) {
    let n = rows * cols;
    let mut g = Graph::with_capacity(n, 2 * n);
    g.add_nodes(n);
    let idx = |r: usize, c: usize| NodeId((r * cols + c) as u32);
    let mut rng = seed;
    let next = |r: &mut u64| -> u64 {
        *r ^= *r << 13;
        *r ^= *r >> 7;
        *r ^= *r << 17;
        *r
    };
    for r in 0..rows {
        for c in 0..cols {
            let w = |rng: &mut u64| (next(rng) % (2 * w_max as u64 + 1)) as i64 - w_max;
            if c + 1 < cols {
                g.add_edge(idx(r, c), idx(r, c + 1), w(&mut rng));
            }
            if r + 1 < rows {
                g.add_edge(idx(r, c), idx(r + 1, c), w(&mut rng));
            }
        }
    }
    (g, NodeId(0))
}

// ── Benchmark groups ──────────────────────────────────────────────────────────

fn bench_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_graph");
    // HJS is slow (ms range) — use Flat sampling so criterion takes a fixed
    // number of independent samples instead of trying to pack many iterations
    // into each measurement window.
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n in &[50usize, 100, 200, 400, 800] {
        let (g, src) = make_sparse_graph(n, 10, 42);

        group.bench_with_input(BenchmarkId::new("HJS", n), &(&g, src), |b, &(g, src)| {
            b.iter(|| sssp(black_box(g), black_box(src)))
        });
        group.bench_with_input(
            BenchmarkId::new("HJS-forced", n),
            &(&g, src),
            |b, &(g, src)| b.iter(|| sssp_hjs_forced(black_box(g), black_box(src), 2)),
        );
        group.bench_with_input(
            BenchmarkId::new("Goldberg", n),
            &(&g, src),
            |b, &(g, src)| b.iter(|| goldberg(black_box(g), black_box(src))),
        );
        group.bench_with_input(
            BenchmarkId::new("BellmanFord", n),
            &(&g, src),
            |b, &(g, src)| b.iter(|| bellman_ford(black_box(g), black_box(src))),
        );
    }
    group.finish();
}

fn bench_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_graph");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n in &[200usize, 500, 1000] {
        let (g, src) = make_path_graph(n, 5, 99);

        group.bench_with_input(BenchmarkId::new("HJS", n), &(&g, src), |b, &(g, src)| {
            b.iter(|| sssp(black_box(g), black_box(src)))
        });
        group.bench_with_input(
            BenchmarkId::new("HJS-forced", n),
            &(&g, src),
            |b, &(g, src)| b.iter(|| sssp_hjs_forced(black_box(g), black_box(src), 2)),
        );
        group.bench_with_input(
            BenchmarkId::new("Goldberg", n),
            &(&g, src),
            |b, &(g, src)| b.iter(|| goldberg(black_box(g), black_box(src))),
        );
        group.bench_with_input(
            BenchmarkId::new("BellmanFord", n),
            &(&g, src),
            |b, &(g, src)| b.iter(|| bellman_ford(black_box(g), black_box(src))),
        );
    }
    group.finish();
}

fn bench_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_graph");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &(r, c_) in &[(10usize, 10usize), (15, 15), (20, 20)] {
        let label = r * c_;
        let (g, src) = make_grid_graph(r, c_, 3, 7);

        group.bench_with_input(
            BenchmarkId::new("HJS", label),
            &(&g, src),
            |b, &(g, src)| b.iter(|| sssp(black_box(g), black_box(src))),
        );
        group.bench_with_input(
            BenchmarkId::new("HJS-forced", label),
            &(&g, src),
            |b, &(g, src)| b.iter(|| sssp_hjs_forced(black_box(g), black_box(src), 2)),
        );
        group.bench_with_input(
            BenchmarkId::new("Goldberg", label),
            &(&g, src),
            |b, &(g, src)| b.iter(|| goldberg(black_box(g), black_box(src))),
        );
        group.bench_with_input(
            BenchmarkId::new("BellmanFord", label),
            &(&g, src),
            |b, &(g, src)| b.iter(|| bellman_ford(black_box(g), black_box(src))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_sparse, bench_path, bench_grid);
criterion_main!(benches);
