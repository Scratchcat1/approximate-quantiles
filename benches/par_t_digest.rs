use approximate_quantiles::t_digest::{
    par_t_digest::ParTDigest,
    scale_functions::{inv_k1, k1},
    t_digest::TDigest,
};
use approximate_quantiles::traits::Digest;
use approximate_quantiles::util::{gen_asc_vec, gen_uniform_vec};
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use std::mem;

fn par_t_digest_add_buffer_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("par_t_digest_add_buffer_in_order_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes(
            *size as u64 * mem::size_of::<f64>() as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_vec(size);
            b.iter(|| {
                let mut digest =
                    ParTDigest::new(10000, 50000, &|| TDigest::new(&k1, &inv_k1, 50.0));
                digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

fn par_t_digest_add_buffer_uniform_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("par_t_digest_add_buffer_uniform_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes(
            *size as u64 * mem::size_of::<f64>() as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut digest =
                    ParTDigest::new(10000, 50000, &|| TDigest::new(&k1, &inv_k1, 50.0));
                digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    par_t_digest_add_buffer_in_order_range,
    par_t_digest_add_buffer_uniform_range
);
criterion_main!(benches);
