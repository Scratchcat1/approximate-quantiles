use approximate_quantiles::buffered_digest::BufferedDigest;
use approximate_quantiles::t_digest::{
    scale_functions::{inv_k1, k1},
    t_digest::TDigest,
};
use approximate_quantiles::traits::Digest;
use approximate_quantiles::util::{gen_asc_vec, gen_uniform_vec};
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};

fn buffered_t_digest_add_buffer_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("buffered_t_digest_add_buffer_in_order_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_vec(size);
            b.iter(|| {
                let inner_digest = TDigest::new(&k1, &inv_k1, black_box(3000.0));
                let mut buffered_digest = BufferedDigest::new(inner_digest, 40_000);
                buffered_digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

fn buffered_t_digest_add_buffer_uniform_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("buffered_t_digest_add_buffer_uniform_range");
    group.plot_config(plot_config);

    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let inner_digest = TDigest::new(&k1, &inv_k1, black_box(3000.0));
                let mut buffered_digest = BufferedDigest::new(inner_digest, 40_000);
                buffered_digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    buffered_t_digest_add_buffer_in_order_range,
    buffered_t_digest_add_buffer_uniform_range,
);
criterion_main!(benches);
