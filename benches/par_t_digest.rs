use approximate_quantiles::t_digest::{
    par_t_digest::ParTDigest,
    scale_functions::{inv_k1, k1},
    t_digest::TDigest,
};
use approximate_quantiles::traits::Digest;
use approximate_quantiles::util::gen_asc_vec;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn par_t_digest_add_buffer_in_order(c: &mut Criterion) {
    c.bench_function("par_t_digest_add_buffer_in_order_single", |b| {
        let mut digest = ParTDigest::new(10000, 50000, &|| TDigest::new(&k1, &inv_k1, 50.0));
        let mut input = 0..;
        b.iter(|| {
            digest.add_buffer(vec![black_box(input.next().unwrap() as f64)]);
        })
    });
    c.bench_function("par_t_digest_add_buffer_in_order_100", |b| {
        b.iter(|| {
            let buffer = gen_asc_vec(100);
            let mut digest = ParTDigest::new(10000, 50000, &|| TDigest::new(&k1, &inv_k1, 50.0));
            digest.add_buffer(buffer);
        })
    });

    c.bench_function("par_t_digest_add_buffer_in_order_10000", |b| {
        b.iter(|| {
            let buffer = gen_asc_vec(10_000);
            let mut digest = ParTDigest::new(10000, 50000, &|| TDigest::new(&k1, &inv_k1, 50.0));
            digest.add_buffer(buffer);
        })
    });

    c.bench_function("par_t_digest_add_buffer_in_order_100000", |b| {
        b.iter(|| {
            let buffer = gen_asc_vec(100_000);
            let mut digest = ParTDigest::new(10000, 50000, &|| TDigest::new(&k1, &inv_k1, 50.0));
            digest.add_buffer(buffer);
        })
    });
}

criterion_group!(benches, par_t_digest_add_buffer_in_order);
criterion_main!(benches);
