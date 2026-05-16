use std::time::Duration;

use criterion::{BenchmarkId, Criterion, SamplingMode, black_box, criterion_group, criterion_main};
use hjs_sssp::generators::{make_grid_graph, make_path_graph, make_random_graph};
use hjs_sssp::sssp::{bellman_ford, goldberg, sssp, sssp_hjs_forced};

// Bellman-Ford is O(V·E).  At 4n edges that is O(4n²), so it is only included
// for graphs small enough that it finishes in a reasonable time.
const BF_NODE_LIMIT: usize = 2_000;
// sssp_hjs_forced fires the full path-cover recursion at every level; useful
// for small graphs but redundant overhead at larger sizes.
const HJS_FORCED_LIMIT: usize = 2_000;

// ── Helpers ───────────────────────────────────────────────────────────────────

macro_rules! bench_all {
    ($group:expr, $label:expr, $g:expr, $src:expr, $forced:expr, $bf:expr) => {{
        let (g, src) = ($g, $src);
        $group.bench_with_input(
            BenchmarkId::new("HJS", $label),
            &(&g, src),
            |b, &(g, src)| b.iter(|| sssp(black_box(g), black_box(src))),
        );
        if $forced {
            $group.bench_with_input(
                BenchmarkId::new("HJS-forced", $label),
                &(&g, src),
                |b, &(g, src)| b.iter(|| sssp_hjs_forced(black_box(g), black_box(src), 2)),
            );
        }
        $group.bench_with_input(
            BenchmarkId::new("Goldberg", $label),
            &(&g, src),
            |b, &(g, src)| b.iter(|| goldberg(black_box(g), black_box(src))),
        );
        if $bf {
            $group.bench_with_input(
                BenchmarkId::new("BellmanFord", $label),
                &(&g, src),
                |b, &(g, src)| b.iter(|| bellman_ford(black_box(g), black_box(src))),
            );
        }
    }};
}

// ── Benchmark groups ──────────────────────────────────────────────────────────

/// Random sparse graphs (sprand): n nodes, 4n arcs, w_max = n.
fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("sprand");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    for &n in &[1_000usize, 10_000, 100_000] {
        let (g, src) = make_random_graph(n, 4 * n, n as i64, 42);
        bench_all!(group, n, g, src, n <= HJS_FORCED_LIMIT, n <= BF_NODE_LIMIT);
    }
    group.finish();
}

/// Grid graphs (spgrid): rows×cols nodes, w_max = n.
/// Sizes: 32×32 ≈ 1 K, 100×100 = 10 K, 316×316 ≈ 100 K nodes.
fn bench_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("spgrid");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    for &(rows, cols) in &[(32usize, 32usize), (100, 100), (316, 316)] {
        let n = rows * cols;
        let (g, src) = make_grid_graph(rows, cols, n as i64, 7);
        bench_all!(group, n, g, src, n <= HJS_FORCED_LIMIT, n <= BF_NODE_LIMIT);
    }
    group.finish();
}

/// Path graphs (sppath): n nodes, w_max = n.
fn bench_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("sppath");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    for &n in &[1_000usize, 10_000, 100_000] {
        let (g, src) = make_path_graph(n, n as i64, 99);
        bench_all!(group, n, g, src, n <= HJS_FORCED_LIMIT, n <= BF_NODE_LIMIT);
    }
    group.finish();
}

criterion_group!(benches, bench_random, bench_grid, bench_path);
criterion_main!(benches);
