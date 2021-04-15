use approximate_quantiles::buffered_digest::BufferedDigest;
use approximate_quantiles::parallel_digest::ParallelDigest;
use approximate_quantiles::t_digest::{
    centroid::Centroid,
    par_t_digest::ParTDigest,
    scale_functions,
    scale_functions::{inv_k1, k1},
    t_digest::TDigest,
};
use approximate_quantiles::traits::Digest;
use approximate_quantiles::util::{
    gen_asc_centroid_vec, gen_uniform_centroid_vec, gen_uniform_vec,
};
use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rayon;

fn t_digest_add_buffer_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_add_buffer_in_order_range");
    group.plot_config(plot_config);
    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_centroid_vec(size);
            b.iter(|| {
                let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
                digest.add_centroid_buffer(test_input.clone());
            });
        });
    }
    group.finish();
}

fn t_digest_add_buffer_uniform_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_add_buffer_uniform_range");
    group.plot_config(plot_config);

    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_centroid_vec(size);
            b.iter(|| {
                let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
                digest.add_centroid_buffer(test_input.clone());
            });
        });
    }
    group.finish();
}

fn t_digest_add_cluster_in_order_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_add_cluster_in_order_range");
    group.plot_config(plot_config);

    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_asc_centroid_vec(size);
            b.iter(|| {
                let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
                digest.add_cluster(test_input.clone(), 5.0);
            });
        });
    }
    group.finish();
}

fn t_digest_add_cluster_uniform_range(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_add_cluster_uniform_range");
    group.plot_config(plot_config);

    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_centroid_vec(size);
            b.iter(|| {
                let mut digest = TDigest::new(&k1, &inv_k1, black_box(60.0));
                digest.add_cluster(test_input.clone(), 5.0);
            });
        });
    }
    group.finish();
}

fn t_digest_add_cluster_tree_uniform_range(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_add_cluster_tree_uniform_range");
    // group.plot_config(plot_config);

    for size in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let test_input = gen_uniform_centroid_vec(size);
            b.iter(|| {
                let mut digest = TDigest::new(&k1, &inv_k1, black_box(20.0));
                digest.add_cluster_tree(test_input.clone(), 5.0);
            });
        });
    }
    group.finish();
}

fn t_digest_util(c: &mut Criterion) {
    c.bench_function("t_digest_util_total_weight_empty", |b| {
        let digest = TDigest::new(&k1, &inv_k1, 200.0);
        b.iter(|| {
            black_box(digest.total_weight());
        })
    });

    c.bench_function("t_digest_util_total_weight", |b| {
        let buffer = gen_asc_centroid_vec(100_000);
        let mut digest = TDigest::new(&k1, &inv_k1, 200.0);
        digest.add_centroid_buffer(buffer);
        b.iter(|| {
            black_box(digest.total_weight());
        })
    });

    c.bench_function("t_digest_util_find_closest_centroids", |b| {
        let buffer = gen_asc_centroid_vec(100_000);
        let mut digest = TDigest::new(&k1, &inv_k1, 200.0);
        digest.add_centroid_buffer(buffer);
        let mut iterator = 0..;
        b.iter(|| {
            digest.find_closest_centroids(black_box(&Centroid {
                mean: iterator.next().unwrap() as f64,
                weight: 1.0,
            }));
        })
    });

    c.bench_function("t_digest_add_centroids", |b| {
        let buffer = gen_uniform_centroid_vec::<f64>(2);
        let c_a = &buffer[0];
        let c_b = &buffer[1];
        b.iter(|| {
            black_box(black_box(c_a.clone()) + black_box(c_b.clone()));
        })
    });
}

fn t_digest_comparison_uniform_range(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_comparison_uniform_range");
    // group.plot_config(plot_config);

    for size in (0..20).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("add_buf_1_thread", size),
            &size,
            |b, &size| {
                let test_input = gen_uniform_centroid_vec(size);
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                pool.install(|| {
                    b.iter(|| {
                        let mut digest = TDigest::new(&k1, &inv_k1, black_box(3000.0));
                        digest.add_centroid_buffer(test_input.clone());
                    });
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("add_buffer", size), &size, |b, &size| {
            let test_input = gen_uniform_centroid_vec(size);
            b.iter(|| {
                let mut digest = TDigest::new(&k1, &inv_k1, black_box(3000.0));
                digest.add_centroid_buffer(test_input.clone());
            });
        });

        //        group.bench_with_input(BenchmarkId::new("add_cluster", size), &size, |b, &size| {
        //            let test_input = gen_uniform_centroid_vec(size);
        //            b.iter(|| {
        //                let mut digest = TDigest::new(&k1, &inv_k1, black_box(3000.0));
        //                digest.add_cluster(test_input.clone(), 10.0);
        //            });
        //        });

        group.bench_with_input(BenchmarkId::new("buffered", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let inner_digest = TDigest::new(&k1, &inv_k1, black_box(3000.0));
                let mut buffered_digest = BufferedDigest::new(inner_digest, 40_000);
                buffered_digest.add_buffer(&test_input);
            });
        });

        group.bench_with_input(BenchmarkId::new("par_t_digest", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut digest =
                    ParTDigest::new(10_000, 50_000, &|| TDigest::new(&k1, &inv_k1, 3000.0));
                digest.add_buffer(&test_input);
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |b, &size| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let mut digest = ParallelDigest::new(
                    (0..num_cpus::get())
                        .map(|_| TDigest::new(&k1, &inv_k1, black_box(3000.0)))
                        .collect(),
                );
                digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

fn t_digest_add_cluster_comparison_uniform_range(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_add_cluster_comparison_uniform_range");
    // group.plot_config(plot_config);

    for size in (0..20).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("add_cluster", size), &size, |b, &size| {
            let test_input = gen_uniform_centroid_vec(size);
            b.iter(|| {
                let mut digest = TDigest::new(&k1, &inv_k1, black_box(50.0));
                digest.add_cluster(test_input.clone(), 10.0);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("add_cluster_tree", size),
            &size,
            |b, &size| {
                let test_input = gen_uniform_centroid_vec(size);
                b.iter(|| {
                    let mut digest = TDigest::new(&k1, &inv_k1, black_box(50.0));
                    digest.add_cluster_tree(test_input.clone(), 10.0);
                });
            },
        );
    }
    group.finish();
}

fn t_digest_compression_comparison_uniform(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_compression_comparison_uniform_100_000");
    // group.plot_config(plot_config);

    let size = 100_000;
    // Delta is the compression parameter
    for delta in (4..14).map(|x| (1 << x) as f64) {
        group.bench_with_input(
            BenchmarkId::new("add_buffer", delta),
            &delta,
            |b, &delta| {
                let test_input = gen_uniform_centroid_vec(size);
                b.iter(|| {
                    let mut digest = TDigest::new(&k1, &inv_k1, delta);
                    digest.add_centroid_buffer(test_input.clone());
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("buffered", delta), &delta, |b, &delta| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let inner_digest = TDigest::new(&k1, &inv_k1, delta);
                let mut buffered_digest = BufferedDigest::new(inner_digest, 40_000);
                buffered_digest.add_buffer(&test_input);
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", delta), &delta, |b, &delta| {
            let test_input = gen_uniform_vec(size);
            b.iter(|| {
                let digest_func = || TDigest::new(&k1, &inv_k1, delta);
                let mut digest = ParTDigest::new(10_000, 50_000, &digest_func);
                digest.add_buffer(&test_input);
            });
        });
    }
    group.finish();
}

fn t_digest_add_cluster_compression_comparison_uniform(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("t_digest_add_cluster_compression_comparison_uniform_40_000");
    // group.plot_config(plot_config);

    let size = 40_000;
    // Delta is the compression parameter
    for delta in (4..14).map(|x| (1 << x) as f64) {
        group.bench_with_input(
            BenchmarkId::new("add_cluster", delta),
            &delta,
            |b, &delta| {
                let test_input = gen_uniform_centroid_vec(size);
                b.iter(|| {
                    let mut digest = TDigest::new(&k1, &inv_k1, delta);
                    digest.add_cluster(test_input.clone(), 5.0);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("add_cluster_tree", delta),
            &delta,
            |b, &delta| {
                let test_input = gen_uniform_centroid_vec(size);
                b.iter(|| {
                    let mut digest = TDigest::new(&k1, &inv_k1, delta);
                    digest.add_cluster_tree(test_input.clone(), 5.0);
                });
            },
        );
    }
    group.finish();
}

fn t_digest_scale_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("t_digest_scale_functions");
    group.bench_function("t_digest_scale_functions/k0", |b| {
        b.iter(|| {
            black_box(scale_functions::k0(black_box(0.25), 10.0, 100.0));
        })
    });

    group.bench_function("t_digest_scale_functions/k1", |b| {
        b.iter(|| {
            black_box(scale_functions::k1(black_box(0.25), 10.0, 100.0));
        })
    });

    group.bench_function("t_digest_scale_functions/k2", |b| {
        b.iter(|| {
            black_box(scale_functions::k2(black_box(0.25), 10.0, 100.0));
        })
    });

    group.bench_function("t_digest_scale_functions/k3", |b| {
        b.iter(|| {
            black_box(scale_functions::k3(black_box(0.25), 10.0, 100.0));
        })
    });

    group.bench_function("t_digest_scale_functions/inv_k0", |b| {
        b.iter(|| {
            black_box(scale_functions::inv_k0(black_box(0.25), 10.0, 100.0));
        })
    });

    group.bench_function("t_digest_scale_functions/inv_k1", |b| {
        b.iter(|| {
            black_box(scale_functions::inv_k1(black_box(0.25), 10.0, 100.0));
        })
    });

    group.bench_function("t_digest_scale_functions/inv_k2", |b| {
        b.iter(|| {
            black_box(scale_functions::inv_k2(black_box(0.25), 10.0, 100.0));
        })
    });

    group.bench_function("t_digest_scale_functions/inv_k3", |b| {
        b.iter(|| {
            black_box(scale_functions::inv_k3(black_box(0.25), 10.0, 100.0));
        })
    });
}

criterion_group!(
    benches,
    t_digest_add_buffer_in_order_range,
    t_digest_add_buffer_uniform_range,
    t_digest_add_cluster_in_order_range,
    t_digest_add_cluster_uniform_range,
    t_digest_add_cluster_tree_uniform_range,
    t_digest_util,
    t_digest_comparison_uniform_range,
    t_digest_add_cluster_comparison_uniform_range,
    t_digest_compression_comparison_uniform,
    t_digest_add_cluster_compression_comparison_uniform,
    t_digest_scale_functions
);
criterion_main!(benches);
