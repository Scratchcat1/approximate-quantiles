use approximate_quantiles::t_digest::{
    scale_functions::{inv_k1, k1},
    Centroid, TDigest,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn t_digest_add_buffer_in_order(c: &mut Criterion) {
    c.bench_function("t_digest_add_buffer_in_order_single", |b| {
        let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
        b.iter(|| {
            digest.add_buffer(vec![Centroid {
                mean: black_box(2.0),
                weight: 1.0,
            }]);
        })
    });
    c.bench_function("t_digest_add_buffer_in_order_100", |b| {
        b.iter(|| {
            let buffer = (0..100)
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_buffer(buffer);
        })
    });

    c.bench_function("t_digest_add_buffer_in_order_10000", |b| {
        b.iter(|| {
            let buffer = (0..10000)
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_buffer(buffer);
        })
    });

    c.bench_function("t_digest_add_buffer_in_order_100000", |b| {
        b.iter(|| {
            let buffer = (0..100000)
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_buffer(buffer);
        })
    });
}

fn t_digest_add_cluster_in_order(c: &mut Criterion) {
    c.bench_function("t_digest_add_cluster_in_order_single", |b| {
        let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
        b.iter(|| {
            digest.add_cluster(
                vec![Centroid {
                    mean: black_box(2.0),
                    weight: 1.0,
                }],
                5.0,
            );
        })
    });
    c.bench_function("t_digest_add_cluster_in_order_100", |b| {
        b.iter(|| {
            let buffer = (0..100)
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_cluster(buffer, 5.0);
        })
    });

    c.bench_function("t_digest_add_cluster_in_order_10000", |b| {
        b.iter(|| {
            let buffer = (0..10000)
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_cluster(buffer, 5.0);
        })
    });

    c.bench_function("t_digest_add_cluster_in_order_100000", |b| {
        b.iter(|| {
            let buffer = (0..100000)
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_cluster(buffer, 5.0);
        })
    });
}

fn reference_vector_pushing(c: &mut Criterion) {
    c.bench_function("vector_push_single", |b| {
        let mut vec = Vec::new();
        b.iter(|| {
            vec.push(black_box(1.0));
        });
    });

    c.bench_function("vector_push_100", |b| {
        b.iter(|| {
            let mut vec = Vec::new();
            for i in 0..100 {
                vec.push(black_box(i));
            }
        });
    });

    c.bench_function("vector_push_10000", |b| {
        b.iter(|| {
            let mut vec = Vec::new();
            for i in 0..10000 {
                vec.push(black_box(i));
            }
        });
    });

    c.bench_function("vector_push_100000", |b| {
        b.iter(|| {
            let mut vec = Vec::new();
            for i in 0..100000 {
                vec.push(black_box(i));
            }
        });
    });
}

criterion_group!(
    benches,
    t_digest_add_buffer_in_order,
    t_digest_add_cluster_in_order,
    reference_vector_pushing
);
criterion_main!(benches);
