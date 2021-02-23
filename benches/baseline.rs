use approximate_quantiles::t_digest::centroid::Centroid;
use approximate_quantiles::util::keyed_sum_tree::KeyedSumTree;
use approximate_quantiles::util::{gen_uniform_centroid_random_weight_vec, gen_uniform_vec};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::prelude::*;

struct IntCentroid {
    pub mean: i64,
    pub weight: i64,
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

    c.bench_function("vector_centroid_add_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid<f64>> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let mut vec = Vec::new();
            let mut x = 0 as f64;
            for c in buffer {
                x += c.mean * c.weight;
                vec.push(x);
            }
            black_box(x);
        });
    });

    c.bench_function("vector_centroid_add_100000-warm", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid<f64>> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let num_elements: f64 = buffer.iter().map(|c| c.weight).sum();
            black_box(num_elements);
            let min = buffer.first().unwrap();
            let max = buffer.last().unwrap();
            black_box(min);
            black_box(max);
            let mut vec: Vec<Centroid<f64>> = Vec::new();
            let mut sum = 0.0;
            let mut weight = 0.0;
            for c in buffer {
                if weight + c.weight >= c.mean * 100.0 {
                    vec.push(Centroid {
                        weight,
                        mean: sum / weight,
                    });
                    weight = 0.0;
                    sum = 0.0;
                } else {
                    weight += c.weight;
                    sum += c.mean * c.weight;
                }
            }
            black_box(vec);
            black_box(sum);
            black_box(weight);
        });
    });

    c.bench_function("vector_int_centroid_add_100000", |b| {
        b.iter(|| {
            let buffer: Vec<IntCentroid> = (0..black_box(100000))
                .map(|x| IntCentroid { mean: x, weight: 1 })
                .collect();
            let mut vec = Vec::new();
            let mut x = 0 as i64;
            for c in buffer {
                x += c.mean * c.weight;
                vec.push(x);
            }
            black_box(vec);
        });
    });

    c.bench_function("mergesort_sum_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid<f64>> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let mut weight = 0.0;
            let mut sum = 0.0;
            for centroid in buffer {
                weight += centroid.weight;
                sum += centroid.weight * centroid.mean;
            }
            assert!(sum > 10.0);
            assert!(weight > 10.0);
        });
    });

    c.bench_function("mergesort_buffered_centroid_sum_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid<f64>> = (0..black_box(100000))
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

                if acc_buffer.len() == 16 {
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

    c.bench_function("mergesort_buffered_dod_centroid_sum_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid<f64>> = (0..black_box(100000))
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
            let means: Vec<f64> = (0..100000).map(|c| c as f64).collect();
            let weights: Vec<f64> = (0..100000).map(|c| c as f64).collect();
            let mut acc_mean_buffer = Vec::new();
            let mut acc_weight_buffer = Vec::new();
            for i in 0..means.len() {
                acc_mean_buffer.push(means[i]);
                acc_weight_buffer.push(weights[i]);

                if acc_mean_buffer.len() == 16 {
                    let mut new_weight = 0.0;
                    let mut new_sum = 0.0;
                    for k in 0..acc_mean_buffer.len() {
                        new_weight += acc_weight_buffer[k];
                    }
                    for k in 0..acc_mean_buffer.len() {
                        new_sum += acc_weight_buffer[k] * acc_mean_buffer[k];
                    }
                    acc = acc
                        + Centroid {
                            mean: new_sum / new_weight,
                            weight: new_weight,
                        };
                    acc_mean_buffer.clear();
                    acc_weight_buffer.clear();
                }
            }
        });
    });

    c.bench_function("mergesort_dod_centroid_sum_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid<f64>> = (0..black_box(100000))
                .map(|x| Centroid {
                    mean: x as f64,
                    weight: 1.0,
                })
                .collect();
            buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let means: Vec<f64> = (0..100000).map(|c| c as f64).collect();
            let weights: Vec<f64> = (0..100000).map(|c| c as f64).collect();
            let mut acc_sum = 0.0;
            let mut acc_weight = 0.0;
            for i in 0..means.len() {
                acc_sum += means[i] * weights[i];
                acc_weight += weights[i];
            }
            assert!(acc_sum >= 0.0);
            assert!(acc_weight >= 0.0);
        });
    });

    c.bench_function("mergesort_sum_centroids_100000", |b| {
        b.iter(|| {
            let mut buffer: Vec<Centroid<f64>> = (0..black_box(100000))
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

fn mergesort_uniform_range(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("mergesort_uniform_range");
    // group.plot_config(plot_config);
    for size in (0..20).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("f64_sort_by", size), &size, |b, &size| {
            let test_input = gen_uniform_vec::<f64>(size);
            b.iter(|| {
                let mut input_copy = test_input.clone();
                input_copy.sort_by(|a, b| a.partial_cmp(&b).unwrap());
                black_box(input_copy);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("f64_par_sort_by", size),
            &size,
            |b, &size| {
                let test_input = gen_uniform_vec::<f64>(size);
                b.iter(|| {
                    let mut input_copy = test_input.clone();
                    input_copy.par_sort_by(|a, b| a.partial_cmp(&b).unwrap());
                    black_box(input_copy);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("f32", size), &size, |b, &size| {
            let test_input = gen_uniform_vec::<f32>(size);
            b.iter(|| {
                let mut input_copy = test_input.clone();
                input_copy.sort_by(|a, b| a.partial_cmp(&b).unwrap());
                black_box(input_copy);
            });
        });

        group.bench_with_input(BenchmarkId::new("u64", size), &size, |b, &size| {
            let test_input = gen_uniform_vec::<f64>(size)
                .into_iter()
                .map(|x| (x * 1000.0) as u64)
                .collect::<Vec<u64>>();
            b.iter(|| {
                let mut input_copy = test_input.clone();
                input_copy.sort();
                black_box(input_copy);
            });
        });

        group.bench_with_input(BenchmarkId::new("u32", size), &size, |b, &size| {
            let test_input = gen_uniform_vec::<f64>(size)
                .into_iter()
                .map(|x| (x * 1000.0) as u32)
                .collect::<Vec<u32>>();
            b.iter(|| {
                let mut input_copy = test_input.clone();
                input_copy.sort();
                black_box(input_copy);
            });
        });
    }
    group.finish();
}

fn keyed_sum_tree_from_vec(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("keyed_sum_tree_from_vec");
    // group.plot_config(plot_config);
    for size in (0..15).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let test_input = gen_uniform_centroid_random_weight_vec::<f64>(size);
            b.iter(|| {
                let tree = KeyedSumTree::from(&test_input[..]);
                black_box(tree);
            });
        });
    }
    group.finish();
}

fn keyed_sum_tree_less_than_sum(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("keyed_sum_tree_less_than_sum");
    // group.plot_config(plot_config);
    for size in (0..15).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let test_input = gen_uniform_centroid_random_weight_vec(size);
            let tree = KeyedSumTree::from(&test_input[..]);
            b.iter(|| {
                black_box(tree.less_than_sum(10000.0));
            });
        });
    }
    group.finish();
}

fn keyed_sum_tree_delete(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("keyed_sum_tree_delete");
    // group.plot_config(plot_config);
    for size in (0..15).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let test_input = gen_uniform_centroid_random_weight_vec::<f64>(size);
            b.iter(|| {
                let mut tree = KeyedSumTree::from(&test_input[..]);
                for input in &test_input {
                    black_box(tree.delete(input.mean));
                }
            });
        });
    }
    group.finish();
}

fn keyed_sum_tree_closest_keys(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("keyed_sum_tree_closest_keys");
    // group.plot_config(plot_config);
    for size in (0..15).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let test_input = gen_uniform_centroid_random_weight_vec(size);
            let tree = KeyedSumTree::from(&test_input[..]);
            b.iter(|| {
                black_box(tree.closest_keys(10000.0));
            });
        });
    }
    group.finish();
}

fn keyed_sum_tree_sorted_vec_key(c: &mut Criterion) {
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("keyed_sum_tree_sorted_vec_key");
    // group.plot_config(plot_config);
    for size in (0..15).map(|x| 1 << x) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let test_input = gen_uniform_centroid_random_weight_vec::<f64>(size);
            let tree = KeyedSumTree::from(&test_input[..]);
            b.iter(|| {
                black_box(tree.sorted_vec_key());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    reference_benchmarks,
    mergesort_uniform_range,
    keyed_sum_tree_from_vec,
    keyed_sum_tree_less_than_sum,
    keyed_sum_tree_delete,
    keyed_sum_tree_closest_keys,
    keyed_sum_tree_sorted_vec_key
);
criterion_main!(benches);
