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

fn t_digest_util(c: &mut Criterion) {
    c.bench_function("t_digest_util_total_weight_empty", |b| {
        let digest = TDigest::new(&k1, &inv_k1, 20.0);
        b.iter(|| {
            black_box(digest.total_weight());
        })
    });

    c.bench_function("t_digest_util_total_weight", |b| {
        let buffer = (0..100000)
            .map(|x| Centroid {
                mean: x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_buffer(buffer);
        b.iter(|| {
            black_box(digest.total_weight());
        })
    });

    c.bench_function("t_digest_util_k_size", |b| {
        let buffer = (0..100000)
            .map(|x| Centroid {
                mean: x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_buffer(buffer);
        let mut iterator = 0..;
        b.iter(|| {
            digest.k_size(black_box(&Centroid {
                mean: iterator.next().unwrap() as f64,
                weight: 1.0,
            }));
        })
    });

    c.bench_function("t_digest_util_find_closest_centroids", |b| {
        let buffer = (0..100000)
            .map(|x| Centroid {
                mean: x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_buffer(buffer);
        let mut iterator = 0..;
        b.iter(|| {
            digest.find_closest_centroids(black_box(&Centroid {
                mean: iterator.next().unwrap() as f64,
                weight: 1.0,
            }));
        })
    });
}

fn reference_benchmarks(c: &mut Criterion) {
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

    c.bench_function("vector_push_add_100000", |b| {
        b.iter(|| {
            let mut vec = Vec::new();
            let mut x = 0 as u128;
            for i in 0..black_box(100000) {
                x += i;
                vec.push(x);
            }
        });
    });

    c.bench_function("mergesort_centroids_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        });
    });

    c.bench_function("mergesort_sum_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let mut acc = 0.0;
            for centroid in buffer {
                acc = acc + centroid.weight;
            }
        });
    });

    c.bench_function("mergesort_buffered_centroid_sum_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let mut acc = Centroid {
                mean: 0.0,
                weight: 0.0,
            };
            let mut acc_buffer = Vec::new();
            for centroid in buffer {
                acc_buffer.push(centroid);

                if acc_buffer.len() == 4 {
                    let mut new_weight = 0.0;
                    let mut new_sum = 0.0;
                    for c in &acc_buffer {
                        new_weight += c.weight;
                        new_sum += c.weight * c.mean;
                    }
                    acc = acc
                        + Centroid {
                            mean: new_sum / new_weight,
                            weight: new_weight,
                        };
                    acc_buffer.clear();
                }
            }
        });
    });

    c.bench_function("mergesort_sum_centroids_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let mut acc = Centroid {
                mean: 0.0,
                weight: 0.0,
            };
            for centroid in buffer {
                acc = acc + centroid;
            }
        });
    });
}

criterion_group!(
    benches,
    t_digest_add_buffer_in_order,
    t_digest_add_cluster_in_order,
    t_digest_util,
    reference_benchmarks
);
criterion_main!(benches);
