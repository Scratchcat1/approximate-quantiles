use approximate_quantiles::t_digest::{
    centroid::Centroid,
    scale_functions::{inv_k1, k1},
    t_digest::TDigest,
};
use approximate_quantiles::util::{gen_asc_centroid_vec, gen_uniform_vec};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn t_digest_add_buffer_in_order(c: &mut Criterion) {
    c.bench_function("t_digest_add_buffer_in_order_single", |b| {
        let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
        b.iter(|| {
            digest.add_centroid_buffer(vec![Centroid {
                mean: black_box(2.0),
                weight: 1.0,
            }]);
        })
    });
    c.bench_function("t_digest_add_buffer_in_order_100", |b| {
        b.iter(|| {
            let buffer = gen_asc_centroid_vec(100);
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_centroid_buffer(buffer);
        })
    });

    c.bench_function("t_digest_add_buffer_in_order_10000", |b| {
        b.iter(|| {
            let buffer = gen_asc_centroid_vec(10_000);
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_centroid_buffer(buffer);
        })
    });

    c.bench_function("t_digest_add_buffer_in_order_100000", |b| {
        b.iter(|| {
            let buffer = gen_asc_centroid_vec(100_000);
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_centroid_buffer(buffer);
        })
    });
}

fn t_digest_add_cluster_in_order(c: &mut Criterion) {
    c.bench_function("t_digest_add_cluster_in_order_single", |b| {
        let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
        let mut iterator = 0..;
        b.iter(|| {
            digest.add_cluster(
                vec![Centroid {
                    mean: black_box(iterator.next().unwrap() as f64),
                    weight: 1.0,
                }],
                5.0,
            );
        })
    });
    c.bench_function("t_digest_add_cluster_in_order_100", |b| {
        b.iter(|| {
            let buffer = gen_asc_centroid_vec(100);
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_cluster(buffer, 5.0);
        })
    });

    c.bench_function("t_digest_add_cluster_in_order_10000", |b| {
        b.iter(|| {
            let buffer = gen_asc_centroid_vec(10_000);
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_cluster(buffer, 5.0);
        })
    });

    c.bench_function("t_digest_add_cluster_in_order_100000", |b| {
        b.iter(|| {
            let buffer = gen_asc_centroid_vec(100_000);
            let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
            digest.add_cluster(buffer, 5.0);
        })
    });
}

fn t_digest_add_cluster_uniform(c: &mut Criterion) {
    c.bench_function("t_digest_add_cluster_uniform_single", |b| {
        let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
        let buffer: Vec<f64> = gen_uniform_vec(10_000_000);
        let mut buff_iter = buffer.into_iter();
        // let mut iterator = 0..;
        b.iter(|| {
            digest.add_cluster(
                vec![Centroid {
                    mean: buff_iter.next().unwrap(),
                    weight: 1.0,
                }],
                5.0,
            );
        });
    });
}

fn t_digest_util(c: &mut Criterion) {
    c.bench_function("t_digest_util_total_weight_empty", |b| {
        let digest = TDigest::new(&k1, &inv_k1, 20.0);
        b.iter(|| {
            black_box(digest.total_weight());
        })
    });

    c.bench_function("t_digest_util_total_weight", |b| {
        let buffer = gen_asc_centroid_vec(100_000);
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_centroid_buffer(buffer);
        b.iter(|| {
            black_box(digest.total_weight());
        })
    });

    c.bench_function("t_digest_util_k_size", |b| {
        let buffer = gen_asc_centroid_vec(100_000);
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_centroid_buffer(buffer);
        let mut iterator = 0..;
        b.iter(|| {
            digest.k_size(black_box(&Centroid {
                mean: iterator.next().unwrap() as f64,
                weight: 1.0,
            }));
        })
    });

    c.bench_function("t_digest_util_find_closest_centroids", |b| {
        let buffer = gen_asc_centroid_vec(100_000);
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_centroid_buffer(buffer);
        let mut iterator = 0..;
        b.iter(|| {
            digest.find_closest_centroids(black_box(&Centroid {
                mean: iterator.next().unwrap() as f64,
                weight: 1.0,
            }));
        })
    });
}

criterion_group!(
    benches,
    t_digest_add_buffer_in_order,
    t_digest_add_cluster_in_order,
    t_digest_add_cluster_uniform,
    t_digest_util,
);
criterion_main!(benches);
