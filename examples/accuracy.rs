use approximate_quantiles::relative_compactor::RCSketch;
use approximate_quantiles::t_digest::{scale_functions, t_digest::TDigest};
use approximate_quantiles::traits::{Digest, OwnedSize};
use approximate_quantiles::util::linear_digest::LinearDigest;
use approximate_quantiles::util::{
    gen_uniform_tan_vec, gen_uniform_vec, opt_accuracy_parameter, sample_digest_accuracy,
};
use num_traits::{Float, NumAssignOps};
use plotters::prelude::*;
use std::error::Error;
use std::path::Path;

pub struct Line<'a, T>
where
    T: Float,
{
    name: String,
    datapoints: Vec<(T, Vec<T>)>,
    colour: &'a RGBColor,
}

pub struct DataStat<T>
where
    T: Float,
{
    x: T,
    y_mean: T,
    y_min: T,
    y_max: T,
}

pub fn plot_line_graph<T>(
    title: &str,
    series: Vec<Line<T>>,
    output_path: &Path,
    x_label: &str,
    y_label: &str,
) -> Result<(), Box<dyn Error>>
where
    T: Float,
{
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

    // println!("{} {} {} {}", min_x, max_x, min_y, max_y);
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .margin_right(30)
        .caption(&title, ("sans-serif", 25))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        // .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            (min_x.to_f64().unwrap()..max_x.to_f64().unwrap()).log_scale(),
            (min_y.to_f64().unwrap() + 1e-4..max_y.to_f64().unwrap()).log_scale(),
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
        let data_stats: Vec<DataStat<T>> = s
            .datapoints
            .iter()
            .map(|outputs| DataStat {
                x: outputs.0,
                y_mean: T::from(outputs.1.iter().map(|x| x.to_f64().unwrap()).sum::<f64>())
                    .unwrap()
                    / T::from(outputs.1.len()).unwrap(),
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
                data_stats.iter().map(|data_stat| {
                    (
                        data_stat.x.to_f64().unwrap(),
                        data_stat.y_mean.to_f64().unwrap(),
                    )
                }),
                s.colour,
            ))
            .expect("Failed to draw mean series")
            .label(&s.name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], s.colour));

        chart
            .draw_series(data_stats.iter().map(|data_stat| {
                PathElement::new(
                    vec![
                        (
                            data_stat.x.to_f64().unwrap(),
                            data_stat.y_max.to_f64().unwrap(),
                        ),
                        (
                            data_stat.x.to_f64().unwrap(),
                            data_stat.y_min.to_f64().unwrap(),
                        ),
                    ],
                    s.colour,
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

fn determine_required_parameter<T>()
where
    T: Float + Send + Sync + NumAssignOps,
{
    let test_func = |digest: &mut dyn Digest<T>, linear_digest: &mut LinearDigest<T>| {
        absolute_error(
            digest.est_quantile_at_value(T::from(1.0).unwrap()),
            linear_digest.est_quantile_at_value(T::from(1.0).unwrap()),
        ) < T::from(1e-4).unwrap()
            && absolute_error(
                digest.est_quantile_at_value(T::from(10.0).unwrap()),
                linear_digest.est_quantile_at_value(T::from(10.0).unwrap()),
            ) < T::from(1e-3).unwrap()
            && absolute_error(
                digest.est_quantile_at_value(T::from(100.0).unwrap()),
                linear_digest.est_quantile_at_value(T::from(100.0).unwrap()),
            ) < T::from(1e-2).unwrap()
            && absolute_error(
                digest.est_quantile_at_value(T::from(250.0).unwrap()),
                linear_digest.est_quantile_at_value(T::from(250.0).unwrap()),
            ) < T::from(1e-1).unwrap()
    };

    let create_rcsketch = |dataset: &[T], param: T| {
        let mut digest = RCSketch::new(dataset.len(), param.to_usize().unwrap());
        digest.add_buffer(dataset);
        digest
    };

    let create_t_digest = |dataset: &[T], param: T| {
        let mut digest = TDigest::new(&scale_functions::k2, &scale_functions::inv_k2, param);
        digest.add_buffer(dataset);
        digest
    };

    let opt_rc_sketch_param = opt_accuracy_parameter(
        create_rcsketch,
        || gen_uniform_tan_vec(100_0000),
        test_func,
        100,
        0.9,
        T::from(1_000.0).unwrap(),
        T::from(2.0).unwrap(),
    )
    .expect("Failed to find solution for compression parameter");
    println!(
        "Required compression parameter for RCSketch was {}, size: {} bytes",
        opt_rc_sketch_param.to_f64().unwrap(),
        {
            let x = create_rcsketch(&gen_uniform_tan_vec(100_0000), opt_rc_sketch_param);
            x.owned_size()
        }
    );

    let opt_t_digest_param = opt_accuracy_parameter(
        create_t_digest,
        || gen_uniform_tan_vec(100_0000),
        test_func,
        100,
        0.9,
        T::from(10_000.0).unwrap(),
        T::from(2.0).unwrap(),
    )
    .expect("Failed to find solution for compression parameter");
    println!(
        "Required compression parameter for T Digest was {}, size: {} bytes",
        opt_t_digest_param.to_f64().unwrap(),
        {
            let x = create_t_digest(&gen_uniform_tan_vec(100_0000), opt_t_digest_param);
            x.owned_size()
        }
    );
}

// fn create_rcsketch<'a, T>(accuracy_param: T) -> Box<dyn Fn(&[T]) -> RCSketch<T> + 'a>
// where
//     T: Float + 'a,
// {
//     Box::new(move |dataset: &[T]| {
//         let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
//         digest.add_buffer(dataset);
//         digest
//     })
// }

// fn create_t_digest<'a, F, G, T>(compression_param: T) -> Box<dyn Fn(&[T]) -> TDigest<F, G, T> + 'a>
// where
//     T: Float + Send + Sync + NumAssignOps + 'a,
//     F: Fn(T, T, T) -> T,
//     G: Fn(T, T, T) -> T,
// {
//     Box::new(move |dataset: &[T]| {
//         let mut digest = TDigest::new(
//             scale_functions::k2,
//             scale_functions::inv_k2,
//             compression_param,
//         );
//         digest.add_buffer(dataset);
//         digest
//     })
// }

fn error_against_quantile<T>()
where
    T: Float + Send + Sync + NumAssignOps,
{
    let test_func =
        |quantile: T| move |digest: &mut dyn Digest<T>| digest.est_value_at_quantile(quantile);

    let create_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_t_digest = |compression_param: T| {
        move |dataset: &[T]| {
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
    for i in &[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5] {
        let accuracy_measurements = sample_digest_accuracy(
            create_rcsketch(T::from(50.0).unwrap()),
            || gen_uniform_vec(100_000),
            test_func(T::from(*i).unwrap()),
            |a, b| absolute_error(a, b) * T::from(1e6).unwrap(),
            100,
        )
        .unwrap();
        s.push((T::from(*i).unwrap(), accuracy_measurements));
    }

    series.push(Line {
        name: "RCSketch".to_string(),
        datapoints: s,
        colour: &RED,
    });

    let mut s = Vec::new();
    for i in &[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5] {
        let accuracy_measurements = sample_digest_accuracy(
            create_t_digest(T::from(100.0).unwrap()),
            || gen_uniform_vec(100_000),
            test_func(T::from(*i).unwrap()),
            |a, b| absolute_error(a, b) * T::from(1e6).unwrap(),
            100,
        )
        .unwrap();
        s.push((T::from(*i).unwrap(), accuracy_measurements));
    }

    series.push(Line {
        name: "t-Digest".to_string(),
        datapoints: s,
        colour: &BLUE,
    });

    plot_line_graph(
        "Error against input for value estimate at quantile",
        series,
        &Path::new("plots/acc_vs_input_est_value_from_quantile.png"),
        "Quantile",
        "Absolute Error (ppm)",
    )
    .unwrap();
}

fn plot_memory_usage_against_compression_parameter<T>()
where
    T: Float + Sync + Send + NumAssignOps,
{
    let create_rcsketch = |dataset: &[T], param: T| {
        let mut digest = RCSketch::new(dataset.len(), param.to_usize().unwrap());
        digest.add_buffer(dataset);
        digest
    };

    let create_t_digest = |dataset: &[T], param: T| {
        let mut digest = TDigest::new(&scale_functions::k2, &scale_functions::inv_k2, param);
        digest.add_buffer(dataset);
        digest
    };

    let mut series = Vec::new();
    let comp_params: Vec<u32> = (0..12).map(|x| 10 << x).collect();
    let mut s = Vec::new();
    for comp_param in &comp_params {
        let x = create_rcsketch(&gen_uniform_vec(100_000), T::from(*comp_param).unwrap());
        println!("{}", x.owned_size());
        s.push((
            T::from(*comp_param).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: "RCSketch".to_string(),
        datapoints: s,
        colour: &RED,
    });

    let mut s = Vec::new();
    for comp_param in &comp_params {
        let x = create_rcsketch(&gen_uniform_vec(10000), T::from(*comp_param).unwrap());
        println!("{}", x.owned_size());
        s.push((
            T::from(*comp_param).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: "RCSketch n = 10^4".to_string(),
        datapoints: s,
        colour: &RED,
    });

    let mut s = Vec::new();
    for comp_param in &comp_params {
        let x = create_t_digest(&gen_uniform_vec(100_000), T::from(*comp_param).unwrap());
        println!("{}", x.owned_size());
        s.push((
            T::from(*comp_param).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: "T-Digest".to_string(),
        datapoints: s,
        colour: &BLUE,
    });
    plot_line_graph(
        "Memory usage against compression/accuracy parameter",
        series,
        &Path::new("plots/mem_vs_comp_param.png"),
        "Compression/Accuracy parameter",
        "Memory usage bytes",
    )
    .unwrap();
}

fn absolute_error<T>(measured: T, actual: T) -> T
where
    T: Float,
{
    // println!("{} {}", measured, actual);
    (measured - actual).abs()
}

fn main() {
    error_against_quantile::<f64>();
    determine_required_parameter::<f32>();
    determine_required_parameter::<f64>();
    plot_memory_usage_against_compression_parameter::<f32>();
    println!("Complete");
}
