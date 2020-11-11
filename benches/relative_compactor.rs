use approximate_quantiles::relative_compactor::RCSketch;
use approximate_quantiles::traits::Digest;
use approximate_quantiles::util::{gen_asc_vec, gen_uniform_vec};
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use std::mem;

fn relative_compactor_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_in_order_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes(
            *size as u64 * mem::size_of::<f64>() as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(black_box(64));
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
        group.throughput(Throughput::Bytes(
            *size as u64 * mem::size_of::<f64>() as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut sketch = RCSketch::new(black_box(64));
                test_input.iter().map(|x| sketch.add(*x)).for_each(drop);
            });
        });
    }
    group.finish();
}

fn relative_compactor_batch_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_batch_in_order_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes(
            *size as u64 * mem::size_of::<f64>() as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_vec(size);
            b.iter(|| {
                let mut digest = RCSketch::new(black_box(64));
                digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

fn relative_compactor_batch_uniform_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("relative_compactor_batch_uniform_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Bytes(
            *size as u64 * mem::size_of::<f64>() as u64,
        ));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut digest = RCSketch::new(black_box(64));
                digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

// fn relative_compactor_in_order(c: &mut Criterion) {
//     c.bench_function("relative_compactor_in_order_single", |b| {
//         let mut sketch = RCSketch::new(black_box(64));
//         b.iter(|| {
//             sketch.add(black_box(1.0));
//         })
//     });
//     c.bench_function("relative_compactor_in_order_100", |b| {
//         b.iter(|| {
//             let mut sketch = RCSketch::new(black_box(64));
//             (0..100).map(|x| sketch.add(x as f64)).for_each(drop);
//         })
//     });

//     c.bench_function("relative_compactor_in_order_10000", |b| {
//         b.iter(|| {
//             let mut sketch = RCSketch::new(black_box(64));
//             (0..10000).map(|x| sketch.add(x as f64)).for_each(drop);
//         })
//     });

//     c.bench_function("relative_compactor_in_order_100000", |b| {
//         b.iter(|| {
//             let mut sketch = RCSketch::new(black_box(64));
//             (0..100000).map(|x| sketch.add(x as f64)).for_each(drop);
//         })
//     });

//     c.bench_function("relative_compactor_batch_in_order_single", |b| {
//         let mut sketch = RCSketch::new(black_box(64));
//         b.iter(|| {
//             sketch.add_buffer(&vec![black_box(1.0)]);
//         })
//     });
//     c.bench_function("relative_compactor_batch_in_order_100", |b| {
//         b.iter(|| {
//             let mut sketch = RCSketch::new(black_box(64));
//             sketch.add_buffer(&gen_asc_vec(100));
//         })
//     });

//     c.bench_function("relative_compactor_batch_in_order_10000", |b| {
//         b.iter_batched(
//             || gen_asc_vec(10_000),
//             |test_input| {
//                 let mut sketch = RCSketch::new(black_box(64));
//                 sketch.add_buffer(&test_input);
//             },
//             BatchSize::SmallInput,
//         )
//     });

//     c.bench_function("relative_compactor_batch_in_order_100000", |b| {
//         b.iter_batched(
//             || gen_asc_vec(100_000),
//             |test_input| {
//                 let mut sketch = RCSketch::new(black_box(64));
//                 sketch.add_buffer(&test_input);
//             },
//             BatchSize::SmallInput,
//         )
//     });
// }

criterion_group!(
    benches,
    relative_compactor_in_order_range,
    relative_compactor_uniform_range,
    relative_compactor_batch_in_order_range,
    relative_compactor_batch_uniform_range
);
criterion_main!(benches);
