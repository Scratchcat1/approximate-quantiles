use approximate_quantiles::relative_compactor::RCSketch;
use approximate_quantiles::traits::Digest;
use approximate_quantiles::util::{gen_asc_vec, gen_uniform_vec};
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};

fn relative_compactor_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_in_order_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                test_input.iter().map(|x| sketch.add(*x)).for_each(drop);
            });
        });
    }
    group.finish();
}

fn relative_compactor_uniform_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_uniform_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                test_input.iter().map(|x| sketch.add(*x)).for_each(drop);
            });
        });
    }
    group.finish();
}

fn relative_compactor_fast_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_fast_in_order_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                sketch.add_buffer_fast(&test_input);
            });
        });
    }
    group.finish();
}

fn relative_compactor_fast_uniform_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_fast_uniform_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                sketch.add_buffer_fast(&test_input);
            });
        });
    }
    group.finish();
}

fn relative_compactor_comparison_uniform_range(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_comparison_uniform_range");
    // group.plot_config(plot_config);
    for size in (0..20).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("default", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                test_input.iter().map(|x| sketch.add(*x)).for_each(drop);
            });
        });
        group.bench_with_input(BenchmarkId::new("buffer", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                sketch.add_buffer(&test_input)
            });
        });
        group.bench_with_input(BenchmarkId::new("buffer_fast", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                sketch.add_buffer_fast(&test_input);
            });
        });
    }
    group.finish();
}

fn relative_compactor_compression_comparison_uniform_range(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_compression_comparison_uniform_range");
    // group.plot_config(plot_config);
    for size in (10..20).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("k-16", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 16);
                test_input.iter().map(|x| sketch.add(*x)).for_each(drop);
            });
        });
        group.bench_with_input(BenchmarkId::new("k-128", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 128);
                sketch.add_buffer(&test_input)
            });
        });
        group.bench_with_input(BenchmarkId::new("k-512", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(size as usize, 512);
                sketch.add_buffer_fast(&test_input);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    relative_compactor_in_order_range,
    relative_compactor_uniform_range,
    relative_compactor_fast_in_order_range,
    relative_compactor_fast_uniform_range,
    relative_compactor_comparison_uniform_range,
    relative_compactor_compression_comparison_uniform_range
);
criterion_main!(benches);
