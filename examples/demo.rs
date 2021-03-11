mod accuracy;
use accuracy::{plot_line_graph, Line, T_DIGEST_CHUNK_SIZE};
use approximate_quantiles::rc_sketch::rc_sketch::RCSketch;
use approximate_quantiles::t_digest::{scale_functions, t_digest::TDigest};
use approximate_quantiles::traits::{Digest, OwnedSize};
use approximate_quantiles::util::linear_digest::LinearDigest;
use approximate_quantiles::util::{gen_uniform_vec, sample_digest_accuracy};
use plotters::prelude::*;
use std::path::Path;

fn main() {
    let dataset_func = |input_size| {
        gen_uniform_vec(input_size)
            .into_iter()
            .map(|x: f32| x.sin().abs())
            .collect::<Vec<f32>>()
    };

    let test_func =
        |quantile: f32| move |digest: &mut dyn Digest<f32>| digest.est_value_at_quantile(quantile);

    let error_func = |a, b| a;

    let create_rcsketch = |accuracy_param: f32| {
        move |dataset: &[f32]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param as usize);
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_linear_digest = move |dataset: &[f32]| {
        let mut digest = LinearDigest::new();
        digest.add_buffer(dataset);
        digest
    };

    let create_t_digest = |compression_param: f32| {
        move |dataset: &[f32]| {
            let mut digest = TDigest::new(
                &scale_functions::k2,
                &scale_functions::inv_k2,
                compression_param,
            );
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let rcsketch_param = 4.0;
    let t_digest_param = 1200.0;

    let input_size = 100_000;
    let quantiles = [
        1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4,
        7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2,
        4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1,
        0.2, 0.4, 0.6, 0.8, 1.0,
    ];
    let rc_sketch_mem_size = {
        let digest = create_rcsketch(rcsketch_param)(&gen_uniform_vec(input_size));
        digest.owned_size()
    };
    let t_digest_mem_size = {
        let digest = create_t_digest(t_digest_param)(&gen_uniform_vec(input_size));
        digest.owned_size()
    };
    let dataset = dataset_func(input_size);

    let mut series = Vec::new();

    let mut s = Vec::new();
    for i in &quantiles {
        let accuracy_measurements = sample_digest_accuracy(
            create_linear_digest,
            || dataset.clone(),
            test_func(*i),
            error_func,
            1,
        )
        .unwrap();
        s.push((*i, accuracy_measurements));
    }

    series.push(Line {
        name: "Actual distribution".to_string(),
        datapoints: s,
        colour: &BLACK,
        marker: None,
    });

    let mut s = Vec::new();
    for i in &quantiles {
        let accuracy_measurements = sample_digest_accuracy(
            create_rcsketch(rcsketch_param),
            || dataset.clone(),
            test_func(*i),
            error_func,
            1,
        )
        .unwrap();
        s.push((*i, accuracy_measurements));
    }

    series.push(Line {
        name: format!("RC Sketch ({} bytes)", rc_sketch_mem_size),
        datapoints: s,
        colour: &RED,
        marker: None,
    });

    let mut s = Vec::new();
    for i in &quantiles {
        let accuracy_measurements = sample_digest_accuracy(
            create_t_digest(t_digest_param),
            || dataset.clone(),
            test_func(*i),
            error_func,
            1,
        )
        .unwrap();
        s.push((*i, accuracy_measurements));
    }

    series.push(Line {
        name: format!("t-Digest ({} bytes)", t_digest_mem_size),
        datapoints: s,
        colour: &BLUE,
        marker: None,
    });

    plot_line_graph(
        &format!("Comparing actual value to estimated",),
        series,
        &Path::new("plots/demo.png"),
        "Quantile",
        "Estimated value",
        false,
    )
    .unwrap();

    println!("Complete");
}
