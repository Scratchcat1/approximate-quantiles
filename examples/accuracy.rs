use approximate_quantiles::relative_compactor::RCSketch;
use approximate_quantiles::t_digest::{scale_functions, t_digest::TDigest};
use approximate_quantiles::traits::Digest;
use approximate_quantiles::util::linear_digest::LinearDigest;
use approximate_quantiles::util::{
    gen_uniform_vec, opt_accuracy_parameter, sample_digest_accuracy,
};
use plotters::prelude::*;
use std::error::Error;
use std::path::Path;

pub struct Line {
    name: String,
    datapoints: Vec<(f64, Vec<f64>)>,
}

pub struct DataStat {
    x: f64,
    y_mean: f64,
    y_min: f64,
    y_max: f64,
}

pub fn plot_line_graph(
    title: &str,
    series: Vec<Line>,
    output_path: &Path,
    x_label: &str,
    y_label: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;
    let min_x = series
        .iter()
        .map(|s| {
            s.datapoints
                .iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap()
        .0;
    let max_x = series
        .iter()
        .map(|s| {
            s.datapoints
                .iter()
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap()
        .0;

    let min_y = *series
        .iter()
        .map(|s| {
            s.datapoints
                .iter()
                .map(|dp| {
                    dp.1.iter()
                        .min_by(|a, b| a.partial_cmp(&b).unwrap())
                        .unwrap()
                })
                .min_by(|a, b| a.partial_cmp(&b).unwrap())
                .unwrap()
        })
        .min_by(|a, b| a.partial_cmp(&b).unwrap())
        .unwrap();

    let max_y = *series
        .iter()
        .map(|s| {
            s.datapoints
                .iter()
                .map(|dp| {
                    dp.1.iter()
                        .max_by(|a, b| a.partial_cmp(&b).unwrap())
                        .unwrap()
                })
                .max_by(|a, b| a.partial_cmp(&b).unwrap())
                .unwrap()
        })
        .max_by(|a, b| a.partial_cmp(&b).unwrap())
        .unwrap();

    println!("{} {} {} {}", min_x, max_x, min_y, max_y);
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .margin_right(30)
        .caption(&title, ("sans-serif", 25))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        // .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            (min_x..max_x).log_scale(),
            (min_y + 1e-4..max_y * 1e6).log_scale(),
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(30)
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    println!("Drawing series");
    series.iter().for_each(|s| {
        let data_stats: Vec<DataStat> = s
            .datapoints
            .iter()
            .map(|outputs| DataStat {
                x: outputs.0,
                y_mean: outputs.1.iter().sum::<f64>() / outputs.1.len() as f64,
                y_min: *outputs
                    .1
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
                y_max: *outputs
                    .1
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
            })
            .collect();

        chart
            .draw_series(LineSeries::new(
                data_stats
                    .iter()
                    .map(|data_stat| (data_stat.x, data_stat.y_mean * 1e6)),
                &BLUE,
            ))
            .expect("Failed to draw mean series")
            .label(&s.name)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(data_stats.iter().map(|data_stat| {
                PathElement::new(
                    vec![
                        (data_stat.x, data_stat.y_max * 1e6),
                        (data_stat.x, data_stat.y_min * 1e6),
                    ],
                    &BLUE,
                )
            }))
            .expect("Failed to draw mean series");
    });

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();

    Ok(())
}

fn determine_required_parameter() {
    let test_func = |digest: &mut dyn Digest, linear_digest: &mut LinearDigest| {
        absolute_error(
            digest.est_quantile_at_value(1.0),
            linear_digest.est_quantile_at_value(1.0),
        ) < 1e-4
            && absolute_error(
                digest.est_quantile_at_value(10.0),
                linear_digest.est_quantile_at_value(10.0),
            ) < 1e-3
            && absolute_error(
                digest.est_quantile_at_value(100.0),
                linear_digest.est_quantile_at_value(100.0),
            ) < 1e-2
            && absolute_error(
                digest.est_quantile_at_value(250.0),
                linear_digest.est_quantile_at_value(250.0),
            ) < 1e-1
    };

    let create_rcsketch = |dataset: &[f64], param: f64| {
        let mut digest = RCSketch::new(dataset.len(), param as usize);
        digest.add_buffer(dataset);
        digest
    };

    let create_t_digest = |dataset: &[f64], param: f64| {
        let mut digest = TDigest::new(&scale_functions::k2, &scale_functions::inv_k2, param);
        digest.add_buffer(dataset);
        digest
    };

    let opt_rc_sketch_param = opt_accuracy_parameter(
        create_rcsketch,
        || gen_uniform_vec(100_000),
        test_func,
        100,
        0.9,
        1_000.0,
        2.0,
    );
    println!(
        "Required compression parameter for RCSketch was {}",
        opt_rc_sketch_param.expect("Failed to find solution for compression parameter")
    );

    let opt_t_digest_param = opt_accuracy_parameter(
        create_t_digest,
        || gen_uniform_vec(100_000),
        test_func,
        100,
        0.9,
        10_000.0,
        2.0,
    );
    println!(
        "Required compression parameter for T Digest was {}",
        opt_t_digest_param.expect("Failed to find solution for compression parameter")
    );
}
fn accuracy_against_space_usage() {
    let test_func = |value: f64| move |digest: &mut dyn Digest| digest.est_quantile_at_value(value);

    let create_rcsketch = |accuracy_param: f64| {
        move |dataset: &[f64]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param as usize);
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_t_digest = |compression_param: f64| {
        move |dataset: &[f64]| {
            let mut digest = TDigest::new(
                &scale_functions::k2,
                &scale_functions::inv_k2,
                compression_param,
            );
            digest.add_buffer(dataset);
            digest
        }
    };

    let mut series = Vec::new();

    let mut s = Vec::new();
    for i in &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
        let accuracy_measurements = sample_digest_accuracy(
            create_rcsketch(50.0),
            || gen_uniform_vec(100_000),
            test_func(*i),
            absolute_error,
            100,
        )
        .unwrap();
        s.push((*i, accuracy_measurements));
    }

    series.push(Line {
        name: "RCSketch".to_string(),
        datapoints: s,
    });

    let mut s = Vec::new();
    for i in &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
        let accuracy_measurements = sample_digest_accuracy(
            create_t_digest(100.0),
            || gen_uniform_vec(100_000),
            test_func(*i),
            absolute_error,
            100,
        )
        .unwrap();
        s.push((*i, accuracy_measurements));
    }

    series.push(Line {
        name: "t-Digest".to_string(),
        datapoints: s,
    });

    plot_line_graph(
        "Error against input for estimate quantile from value",
        series,
        &Path::new("plots/acc_vs_input_est_quantile_from_value.png"),
        "value (0..1000)",
        "Absolute Error (ppm)",
    )
    .unwrap();
}

fn absolute_error(measured: f64, actual: f64) -> f64 {
    // println!("{} {}", measured, actual);
    (measured - actual).abs()
}

fn main() {
    accuracy_against_space_usage();
    determine_required_parameter();
    // plot_line_graph(
    //     "bla",
    //     vec![
    //         Line {
    //             name: "line".to_string(),
    //             datapoints: vec![(0.0, vec![1.0, 2.0, 3.0]), (1.0, vec![1.0, 10.0, 3.0])],
    //         },
    //         Line {
    //             name: "line2".to_string(),
    //             datapoints: vec![(5.0, vec![1.0, -2.0, 3.0]), (3.0, vec![1.0, 10.0, 3.0])],
    //         },
    //     ],
    //     "accuracy",
    //     "storage usage",
    // )
    // .expect("Failed to plot");
    println!("Complete");
}
