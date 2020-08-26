use approximate_quantiles::relative_compactor::RCSketch;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn relative_compactor_in_order(c: &mut Criterion) {
    c.bench_function("relative_compactor_in_order_single", |b| {
        let mut sketch = RCSketch::new(black_box(64));
        b.iter(|| {
            sketch.insert(black_box(1.0));
        })
    });
    c.bench_function("relative_compactor_in_order_100", |b| {
        b.iter(|| {
            let mut sketch = RCSketch::new(black_box(64));
            (0..100).map(|x| sketch.insert(x as f64)).for_each(drop);
        })
    });

    c.bench_function("relative_compactor_in_order_10000", |b| {
        b.iter(|| {
            let mut sketch = RCSketch::new(black_box(64));
            (0..10000).map(|x| sketch.insert(x as f64)).for_each(drop);
        })
    });

    c.bench_function("relative_compactor_in_order_100000", |b| {
        b.iter(|| {
            let mut sketch = RCSketch::new(black_box(64));
            (0..100000).map(|x| sketch.insert(x as f64)).for_each(drop);
        })
    });
}

criterion_group!(benches, relative_compactor_in_order);
criterion_main!(benches);
