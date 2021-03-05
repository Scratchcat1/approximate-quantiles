use approximate_quantiles::parallel_digest::ParallelDigest;
use approximate_quantiles::rc_sketch::{
    compaction_method::CompactionMethod, rc_sketch::RCSketch, rc_sketch2::RCSketch2,
};
use approximate_quantiles::sym_digest::SymDigest;
use approximate_quantiles::t_digest::{scale_functions, t_digest::TDigest};
use approximate_quantiles::traits::{Digest, OwnedSize};
use approximate_quantiles::util::linear_digest::LinearDigest;
use approximate_quantiles::util::{
    gen_growing_blocks_vec, gen_uniform_exp_vec, gen_uniform_tan_vec, gen_uniform_vec,
    opt_accuracy_parameter, sample_digest_accuracy,
};
use num_traits::{Float, NumAssignOps};
use plotters::data::fitting_range;
use plotters::prelude::*;
use std::error::Error;
use std::path::Path;

const T_DIGEST_CHUNK_SIZE: usize = 1_000_000;

pub struct Line<'a, T>
where
    T: Float,
{
    name: String,
    datapoints: Vec<(T, Vec<T>)>,
    colour: &'a RGBColor,
    marker: Option<String>,
}

impl<'a, T> Line<'a, T>
where
    T: Float,
{
    pub fn into_line_f64(self) -> Line<'a, f64> {
        Line {
            name: self.name,
            datapoints: self
                .datapoints
                .into_iter()
                .map(|dp| {
                    (
                        dp.0.to_f64().unwrap(),
                        dp.1.into_iter()
                            .map(|y_val| y_val.to_f64().unwrap())
                            .collect(),
                    )
                })
                .collect(),
            colour: self.colour,
            marker: self.marker,
        }
    }
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

pub fn plot_box_plot_graph<T>(
    title: &str,
    raw_series: Vec<Line<T>>,
    output_path: &Path,
    x_label: &str,
    y_label: &str,
) -> Result<(), Box<dyn Error>>
where
    T: Float,
{
    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();

    root.fill(&WHITE)?;

    let series: Vec<Line<f64>> = raw_series
        .into_iter()
        .map(|series| series.into_line_f64())
        .collect();

    let y_values: Vec<f32> = series
        .iter()
        .map(|s| {
            s.datapoints
                .iter()
                .map(|dp| {
                    let quartiles = Quartiles::new(&dp.1);
                    let outlier_dist = 1.5 * quartiles.values()[3] - quartiles.values()[1];
                    let upper_outlier_boundary = quartiles.values()[3] + outlier_dist;
                    let lower_outlier_boundary = quartiles.values()[1] - outlier_dist;
                    dp.1.iter()
                        .map(|x| *x as f32)
                        .filter(|x| *x <= upper_outlier_boundary && *x >= lower_outlier_boundary)
                        .collect::<Vec<f32>>()
                })
                .flatten()
                .collect::<Vec<f32>>()
        })
        .flatten()
        .collect();
    // println!("{:?}", y_values);
    let y_values_range = fitting_range(y_values.iter().filter(|x| **x <= 1e7));
    let x_values: Vec<f64> = series[0..1]
        .iter()
        .map(|s| s.datapoints.iter().map(|a| a.0).collect::<Vec<f64>>())
        .flatten()
        .collect();

    // println!("{} {} {} {}", min_x, max_x, min_y, max_y);
    // let mut colors = (0..).map(Palette99::pick);
    // println!("{:?}", &y_values_range);
    let mut chart = ChartBuilder::on(&root)
        .margin(15)
        .margin_right(30)
        .caption(&title, ("sans-serif", 25))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        // .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(x_values[..].into_segmented(), y_values_range)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    let mut offsets = (-24..).step_by(24);
    series.iter().for_each(|s| {
        let current_offset = offsets.next().unwrap();
        chart
            .draw_series(s.datapoints.iter().map(|dp| {
                Boxplot::new_vertical(SegmentValue::CenterOf(&dp.0), &Quartiles::new(&dp.1))
                    .width(10)
                    .whisker_width(0.5)
                    .style(s.colour)
                    .offset(current_offset)
            }))
            .expect("Failed to draw boxplot series")
            .label(&s.name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], s.colour));
    });

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();
    println!("Drew {}", title);

    Ok(())
}

pub fn plot_line_graph<T>(
    title: &str,
    mut series: Vec<Line<T>>,
    output_path: &Path,
    x_label: &str,
    y_label: &str,
    show_error_bars: bool,
) -> Result<(), Box<dyn Error>>
where
    T: Float + std::fmt::Debug,
{
    let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
    let marker_size = 14;

    root.fill(&WHITE)?;

    // Ensure the datapoints are in order to avoid the line doubling back.
    series
        .iter_mut()
        .for_each(|s| s.datapoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()));

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
                    // println!(
                    //     "{:?}",
                    //     dp.1.iter()
                    //         .map(|x| x.to_f64().unwrap())
                    //         .collect::<Vec<f64>>()
                    // );
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

        data_stats
            .iter()
            .for_each(|ds| println!("{} {} {:?} {:?}", title, s.name, ds.x, ds.y_mean));

        let line_element = chart
            .draw_series(LineSeries::new(
                data_stats.iter().map(|data_stat| {
                    (
                        data_stat.x.to_f64().unwrap(),
                        data_stat.y_mean.to_f64().unwrap(),
                    )
                }),
                s.colour,
            ))
            .expect("Failed to draw series")
            .label(&s.name);
        match &s.marker {
            Some(marker) => {
                line_element.legend(move |(x, y)| {
                    EmptyElement::at((x, y))
                        + Text::new(
                            marker.clone(),
                            (-marker_size / 2 + 10, -marker_size / 2),
                            TextStyle::from(("sans-serif", marker_size).into_font())
                                .color(s.colour),
                        )
                });
            }
            None => {
                line_element
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], s.colour));
            }
        }
        if let Some(marker) = &s.marker {
            chart
                .draw_series(PointSeries::of_element(
                    data_stats.iter().map(|data_stat| {
                        (
                            data_stat.x.to_f64().unwrap(),
                            data_stat.y_mean.to_f64().unwrap(),
                        )
                    }),
                    marker_size,
                    s.colour,
                    &|coord, size, _| {
                        EmptyElement::at(coord)
                            + Text::new(
                                marker.clone(),
                                (-size / 2, -size / 2),
                                TextStyle::from(("sans-serif", size).into_font()).color(s.colour),
                            )
                    },
                ))
                .expect("Failed to draw points");
        }

        if show_error_bars {
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
        }
    });

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();
    println!("Drew {}", title);

    Ok(())
}

/// Returns each of the distributions to be tested.
/// `returns` Tuples of (distribution name, distribution generation function taking an input size)
fn get_distributions<F>() -> Vec<(String, Box<dyn Fn(i32) -> Vec<F> + Send + Sync>)>
where
    F: Float,
{
    vec![
        (
            "Growing blocks".to_string(),
            Box::new(|input_size| gen_growing_blocks_vec(input_size)),
        ),
        (
            "Reverse exponential".to_string(),
            Box::new(|input_size| gen_uniform_exp_vec(input_size, F::from(1.0).unwrap())),
        ),
        (
            "Uniform".to_string(),
            Box::new(|input_size| gen_uniform_vec(input_size)),
        ),
        (
            "Tan".to_string(),
            Box::new(|input_size| gen_uniform_tan_vec(input_size)),
        ),
        // Made no impact
        // (
        //     "Uniform blocks".to_string(),
        //     Box::new(|input_size| {
        //         gen_uniform_vec::<F>(input_size)
        //             .into_iter()
        //             .map(|x| (x * F::from(1_000_000.0).unwrap()) / F::from(1_000.0).unwrap())
        //             .collect()
        //     }),
        // ),
    ]
}

/// Returns each of the estimation functions to be tested.
/// `returns` Tuples of (estimation function name, estimation function which takes the value to test at and
/// returns a function which takes a digest and returns the estimate at that value,
/// a function which takes a quantile and a sample dataset an outputs the value which the estimation function should be fed)
fn get_estimation_funcs<F>() -> Vec<(
    String,
    Box<dyn Fn(F) -> Box<dyn Fn(&mut dyn Digest<F>) -> F + Send + Sync> + Send + Sync>,
    Box<dyn Fn(F, &[F]) -> F + Send + Sync>,
)>
where
    F: Float + Send + Sync + 'static,
{
    vec![
        (
            "value at quantile".to_string(),
            Box::new(|value: F| {
                Box::new(move |digest: &mut dyn Digest<F>| digest.est_value_at_quantile(value))
            }),
            Box::new(|q, _| q),
        ),
        (
            "quantile at value".to_string(),
            Box::new(|quantile: F| {
                Box::new(move |digest: &mut dyn Digest<F>| digest.est_quantile_at_value(quantile))
            }),
            Box::new(|q, dataset| values_from_quantiles(dataset, &[q])[0]),
        ),
    ]
}

fn determine_required_parameter<T>()
where
    T: Float + Send + Sync + NumAssignOps,
{
    let test_func = |digest: &mut dyn Digest<T>, linear_digest: &mut LinearDigest<T>| {
        let values = [
            linear_digest.est_value_at_quantile(T::from(1e-5).unwrap()),
            linear_digest.est_value_at_quantile(T::from(1e-4).unwrap()),
            linear_digest.est_value_at_quantile(T::from(1e-3).unwrap()),
            linear_digest.est_value_at_quantile(T::from(1e-2).unwrap()),
            linear_digest.est_value_at_quantile(T::from(1e-1).unwrap()),
        ];
        // println!(
        //     "{} {} {}",
        //     values[0].to_f64().unwrap(),
        //     digest.est_quantile_at_value(values[0]).to_f64().unwrap(),
        //     linear_digest
        //         .est_quantile_at_value(values[0])
        //         .to_f64()
        //         .unwrap()
        // );
        absolute_error(
            digest.est_quantile_at_value(values[0]),
            linear_digest.est_quantile_at_value(values[0]),
        ) < T::from(1e-6).unwrap()
            && absolute_error(
                digest.est_quantile_at_value(values[1]),
                linear_digest.est_quantile_at_value(values[1]),
            ) < T::from(1e-5).unwrap()
            && absolute_error(
                digest.est_quantile_at_value(values[2]),
                linear_digest.est_quantile_at_value(values[2]),
            ) < T::from(1e-4).unwrap()
            && absolute_error(
                digest.est_quantile_at_value(values[3]),
                linear_digest.est_quantile_at_value(values[3]),
            ) < T::from(1e-3).unwrap()
    };

    let create_rcsketch = |dataset: &[T], param: T| {
        let mut digest = RCSketch::new(dataset.len(), param.to_usize().unwrap());
        digest.add_buffer(dataset);
        digest
    };

    let create_t_digest = |dataset: &[T], param: T| {
        let mut digest = TDigest::new(&scale_functions::k2, &scale_functions::inv_k2, param);
        dataset
            .chunks(T_DIGEST_CHUNK_SIZE)
            .for_each(|chunk| digest.add_buffer(chunk));
        digest
    };

    // let mut x = gen_uniform_tan_vec::<f32>(100_0000);
    // x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // println!("{:?}", &x[999_990..999_999]);
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

fn values_from_quantiles<T>(dataset: &[T], quantiles: &[T]) -> Vec<T>
where
    T: Float,
{
    let mut digest = LinearDigest::new();
    digest.add_buffer(dataset);
    quantiles
        .iter()
        .map(|quantile| digest.est_value_at_quantile(*quantile))
        .collect()
}

fn value_error_against_quantile<T>()
where
    T: Float + Send + Sync + NumAssignOps + std::fmt::Debug,
{
    let test_func =
        |quantile: T| move |digest: &mut dyn Digest<T>| digest.est_value_at_quantile(quantile);

    let error_func = |a, b| {
        T::max(
            T::from(1.0).unwrap(),
            relative_error(a, b) * T::from(1e6).unwrap(),
        )
    };

    let create_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_rcsketch2 = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch2::new(accuracy_param.to_usize().unwrap());
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_compact_avg_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
            digest.add_buffer_custom(dataset, false, CompactionMethod::AverageNeighbour);
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
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let rcsketch_param = T::from(20.0).unwrap();
    let t_digest_param = T::from(6000.0).unwrap();

    let input_size = 100_000;
    let quantiles = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.4];
    let rc_sketch_mem_size = {
        let digest = create_rcsketch(rcsketch_param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };
    let rc_sketch2_mem_size = {
        let digest = create_rcsketch2(rcsketch_param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };
    let t_digest_mem_size = {
        let digest = create_t_digest(t_digest_param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };

    for (dist_name, dataset_func) in &get_distributions() {
        let mut series = Vec::new();

        let mut s = Vec::new();
        for i in &quantiles {
            let accuracy_measurements = sample_digest_accuracy(
                create_rcsketch(rcsketch_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
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
                create_rcsketch2(rcsketch_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
        }

        series.push(Line {
            name: format!("RC Sketch2 ({} bytes)", rc_sketch2_mem_size),
            datapoints: s,
            colour: &MAGENTA,
            marker: None,
        });

        let mut s = Vec::new();
        for i in &quantiles {
            let accuracy_measurements = sample_digest_accuracy(
                create_compact_avg_rcsketch(rcsketch_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
        }

        series.push(Line {
            name: format!("RC Sketch avg compaction ({} bytes)", rc_sketch_mem_size),
            datapoints: s,
            colour: &YELLOW,
            marker: None,
        });

        let mut s = Vec::new();
        for i in &quantiles {
            let accuracy_measurements = sample_digest_accuracy(
                create_t_digest(t_digest_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
        }

        series.push(Line {
            name: format!("t-Digest ({} bytes)", t_digest_mem_size),
            datapoints: s,
            colour: &BLUE,
            marker: None,
        });

        plot_box_plot_graph(
            &format!(
                "Error against input for value estimate at quantile. Dist: {}",
                dist_name
            ),
            series,
            &Path::new(&format!(
                "plots/acc_vs_input_est_value_from_quantile_{}.png",
                dist_name.to_lowercase().replace(" ", "_")
            )),
            "Quantile",
            "Relative Error (ppm)",
        )
        .unwrap();
    }
}

fn quantile_error_against_value<T>()
where
    T: Float + Send + Sync + NumAssignOps + std::fmt::Debug,
{
    let test_func =
        |value: T| move |digest: &mut dyn Digest<T>| digest.est_quantile_at_value(value);

    let error_func = |a, b| {
        T::max(
            T::from(1.0).unwrap(),
            relative_error(a, b) * T::from(1e6).unwrap(),
        )
    };

    let create_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_rcsketch2 = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch2::new(accuracy_param.to_usize().unwrap());
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_compact_avg_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
            digest.add_buffer_custom(dataset, false, CompactionMethod::AverageNeighbour);
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
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let test_quantiles: Vec<T> = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.2]
        .iter()
        .map(|quantile| T::from(*quantile).unwrap())
        .collect();
    // let mut x = gen_uniform_exp_vec(10000, T::from(1.0).unwrap());
    // x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // println!("{:?}", x);

    let rcsketch_param = T::from(20.0).unwrap();
    let t_digest_param = T::from(3000.0).unwrap();

    let input_size = 100_000;
    let rc_sketch_mem_size = {
        let digest = create_rcsketch(rcsketch_param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };
    let rc_sketch2_mem_size = {
        let digest = create_rcsketch2(rcsketch_param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };
    let t_digest_mem_size = {
        let digest = create_t_digest(t_digest_param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };

    for (dist_name, dataset_func) in &get_distributions() {
        let test_values = values_from_quantiles(&dataset_func(input_size), &test_quantiles);
        println!("{:?}", test_values);
        let mut series = Vec::new();

        let mut s = Vec::new();
        for i in &test_values {
            let accuracy_measurements = sample_digest_accuracy(
                create_rcsketch(rcsketch_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
        }

        series.push(Line {
            name: format!("RCSketch ({} bytes)", rc_sketch_mem_size),
            datapoints: s,
            colour: &RED,
            marker: None,
        });

        let mut s = Vec::new();
        for i in &test_values {
            let accuracy_measurements = sample_digest_accuracy(
                create_rcsketch2(rcsketch_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
        }

        series.push(Line {
            name: format!("RCSketch2 ({} bytes)", rc_sketch2_mem_size),
            datapoints: s,
            colour: &MAGENTA,
            marker: None,
        });

        let mut s = Vec::new();
        for i in &test_values {
            let accuracy_measurements = sample_digest_accuracy(
                create_compact_avg_rcsketch(rcsketch_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
        }

        series.push(Line {
            name: format!("RCSketch avg compaction ({} bytes)", rc_sketch_mem_size),
            datapoints: s,
            colour: &YELLOW,
            marker: None,
        });

        let mut s = Vec::new();
        for i in &test_values {
            let accuracy_measurements = sample_digest_accuracy(
                create_t_digest(t_digest_param),
                || dataset_func(input_size),
                test_func(T::from(*i).unwrap()),
                error_func,
                100,
            )
            .unwrap();
            s.push((T::from(*i).unwrap(), accuracy_measurements));
        }

        series.push(Line {
            name: format!("t-Digest ({} bytes)", t_digest_mem_size),
            datapoints: s,
            colour: &BLUE,
            marker: None,
        });
        plot_box_plot_graph(
            &format!(
                "Error against input for quantile estimate at value. Dist = {}",
                dist_name
            ),
            series,
            &Path::new(&format!(
                "plots/acc_vs_input_est_quantile_from_value_{}.png",
                dist_name.to_lowercase().replace(" ", "_")
            )),
            "Value",
            "Relative Error (ppm)",
        )
        .unwrap();
    }
}

// T Only controls RC Sketch, force 64 bit t digest
fn plot_error_against_mem_usage<T>()
where
    T: Float + Send + Sync + NumAssignOps + std::fmt::Debug,
{
    let test_func =
        |value: T| move |digest: &mut dyn Digest<T>| digest.est_value_at_quantile(value);
    let test_funcf64 =
        |value: f64| move |digest: &mut dyn Digest<f64>| digest.est_value_at_quantile(value);

    let error_func = |a, b| {
        T::max(
            T::from(1.0).unwrap(),
            relative_error(a, b) * T::from(1e6).unwrap(),
        )
    };

    let create_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
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
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let rc_test_values = [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0]
        .iter()
        .map(|x| T::from(*x).unwrap())
        .collect::<Vec<T>>();
    let t_digest_test_values = (4..14)
        // .iter()
        .map(|x| (1 << x) as f64)
        .collect::<Vec<f64>>();

    let quantiles = [
        (1e-5, "X"),
        (1e-4, "▲"),
        (1e-3, "◆"),
        (1e-2, "●"),
        (1e-1, "■"),
    ]
    .iter()
    .map(|(q, marker)| (T::from(*q).unwrap(), marker.to_string()))
    .collect::<Vec<(T, String)>>();

    let input_size = 1_000_000;
    let rc_sketch_mem_size = |param| {
        let digest = create_rcsketch(param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };
    let t_digest_mem_size = |param| {
        let digest = create_t_digest(param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };

    for (dist_name, dataset_func) in &get_distributions() {
        let mut series = Vec::new();

        for (quantile, marker) in &quantiles {
            let mut s = Vec::new();
            for rcsketch_param in &rc_test_values {
                let accuracy_measurements = sample_digest_accuracy(
                    create_rcsketch(*rcsketch_param),
                    || dataset_func(input_size),
                    test_func(*quantile),
                    error_func,
                    100,
                )
                .unwrap();
                s.push((
                    T::from(rc_sketch_mem_size(*rcsketch_param)).unwrap(),
                    accuracy_measurements,
                ));
            }

            series.push(
                Line {
                    name: format!("RCSketch, q = 1e{:?}", quantile.log10().to_i32().unwrap()),
                    datapoints: s,
                    colour: &RED,
                    marker: Some(marker.clone()),
                }
                .into_line_f64(),
            );

            let mut s = Vec::new();
            for t_digest_param in &t_digest_test_values {
                let accuracy_measurements = sample_digest_accuracy(
                    create_t_digest(*t_digest_param),
                    || {
                        dataset_func(input_size)
                            .into_iter()
                            .map(|x| x.to_f64().unwrap())
                            .collect::<Vec<f64>>()
                    },
                    test_funcf64((*quantile).to_f64().unwrap()),
                    |a, b| {
                        error_func(T::from(a).unwrap(), T::from(b).unwrap())
                            .to_f64()
                            .unwrap()
                    },
                    100,
                )
                .unwrap();
                s.push((
                    t_digest_mem_size(*t_digest_param) as f64,
                    accuracy_measurements,
                ));
            }

            series.push(
                Line {
                    name: format!("t-Digest, q = 1e{:?}", quantile.log10().to_i32().unwrap()),
                    datapoints: s,
                    colour: &BLUE,
                    marker: Some(marker.clone()),
                }
                .into_line_f64(),
            );
        }
        plot_line_graph(
            &format!(
                "Error vs memory usage for quantile estimate at value. Dist: {}",
                dist_name
            ),
            series,
            &Path::new(&format!(
                "plots/err_vs_mem_usage_for_est_quantile_from_value_{}.png",
                dist_name.to_lowercase().replace(" ", "_")
            )),
            "Memory (bytes)",
            "Relative Error (ppm)",
            false,
        )
        .unwrap();
    }
}

fn plot_error_against_mem_usage_parallel<T>()
where
    T: Float + Send + Sync + NumAssignOps + std::fmt::Debug + 'static,
{
    let error_func = |a, b| {
        T::max(
            T::from(1.0).unwrap(),
            relative_error(a, b) * T::from(1e6).unwrap(),
        )
    };

    let create_parallel_rcsketch = |threads: usize, accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = ParallelDigest::new(
                (0..threads)
                    .map(|_| {
                        RCSketch::new(dataset.len() / threads, accuracy_param.to_usize().unwrap())
                    })
                    .collect(),
            );
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_parallel_t_digest = |threads: usize, compression_param: T| {
        move |dataset: &[T]| {
            let mut digest = ParallelDigest::new(
                (0..threads)
                    .map(|_| {
                        TDigest::new(
                            &scale_functions::k2,
                            &scale_functions::inv_k2,
                            compression_param / T::from(threads).unwrap(),
                        )
                    })
                    .collect(),
            );
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let rc_test_values = [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0]
        .iter()
        .map(|x| T::from(*x).unwrap())
        .collect::<Vec<T>>();
    let t_digest_test_values = (4..14)
        // .iter()
        .map(|x| T::from(1 << x).unwrap())
        .collect::<Vec<T>>();

    let quantiles = [
        (1e-5, "X"),
        (1e-4, "▲"),
        (1e-3, "◆"),
        (1e-2, "●"),
        (1e-1, "■"),
    ]
    .iter()
    .map(|(q, marker)| (T::from(*q).unwrap(), marker.to_string()))
    .collect::<Vec<(T, String)>>();

    let thread_setups = [(1, &BLUE), (2, &CYAN), (4, &MAGENTA), (8, &RED)];

    let input_size = 1_000_000;
    let rc_sketch_mem_size = |threads, param| {
        let digest = create_parallel_rcsketch(threads, param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };
    let t_digest_mem_size = |threads, param| {
        let digest = create_parallel_t_digest(threads, param)(&gen_uniform_tan_vec(input_size));
        digest.owned_size()
    };

    for (est_name, est_func, gen_est_position) in &get_estimation_funcs() {
        for (dist_name, dataset_func) in &get_distributions() {
            let mut series = Vec::new();

            for (threads, colour) in &thread_setups {
                for (quantile, marker) in &quantiles {
                    let est_value = gen_est_position(*quantile, &dataset_func(input_size));
                    let mut s = Vec::new();
                    for rcsketch_param in &rc_test_values {
                        let accuracy_measurements = sample_digest_accuracy(
                            create_parallel_rcsketch(*threads, *rcsketch_param),
                            || dataset_func(input_size),
                            est_func(est_value),
                            error_func,
                            100,
                        )
                        .unwrap();
                        s.push((
                            T::from(rc_sketch_mem_size(*threads, *rcsketch_param)).unwrap(),
                            accuracy_measurements,
                        ));
                    }

                    series.push(Line {
                        name: format!(
                            "RCSketch, t = {}, q = 1e{:?}",
                            threads,
                            quantile.log10().to_i32().unwrap()
                        ),
                        datapoints: s,
                        colour: colour,
                        marker: Some(marker.clone()),
                    });
                }
            }
            plot_line_graph(
                &format!(
                    "RCSketch error vs memory usage for estimate {} using parallel digest. Dist: {}",
                    est_name, dist_name
                ),
                series,
                &Path::new(&format!(
                    "plots/err_vs_mem_usage_rcsketch_parallel_for_{}_{}.png",
                    est_name.to_lowercase().replace(" ", "_"),
                    dist_name.to_lowercase().replace(" ", "_")
                )),
                "Memory (bytes)",
                "Relative Error (ppm)",
                false,
            )
            .unwrap();
            let mut series = Vec::new();

            for (threads, colour) in &thread_setups {
                for (quantile, marker) in &quantiles {
                    let est_value = gen_est_position(*quantile, &dataset_func(input_size));

                    let mut s = Vec::new();
                    for t_digest_param in &t_digest_test_values {
                        let accuracy_measurements = sample_digest_accuracy(
                            create_parallel_t_digest(*threads, *t_digest_param),
                            || dataset_func(input_size),
                            est_func(est_value),
                            error_func,
                            100,
                        )
                        .unwrap();
                        s.push((
                            T::from(t_digest_mem_size(*threads, *t_digest_param)).unwrap(),
                            accuracy_measurements,
                        ));
                    }

                    series.push(Line {
                        name: format!(
                            "t-Digest, t = {}, q = 1e{:?}",
                            threads,
                            quantile.log10().to_i32().unwrap()
                        ),
                        datapoints: s,
                        colour: colour,
                        marker: Some(marker.clone()),
                    });
                }
            }
            plot_line_graph(
                &format!(
                    "T Digest error vs memory usage for estimate {} using parallel digest. Dist: {}",
                    est_name, dist_name
                ),
                series,
                &Path::new(&format!(
                    "plots/err_vs_mem_usage_tdigest_parallel_for_{}_{}.png",
                    est_name.to_lowercase().replace(" ", "_"),
                    dist_name.to_lowercase().replace(" ", "_")
                )),
                "Memory (bytes)",
                "Relative Error (ppm)",
                false,
            )
            .unwrap();
        }
    }
}

fn plot_error_against_quantiles_full_range<T>()
where
    T: Float + Send + Sync + NumAssignOps + std::fmt::Debug + 'static,
{
    let error_func = |a, b| {
        T::max(
            T::from(1.0).unwrap(),
            relative_error(a, b) * T::from(1e6).unwrap(),
        )
    };

    let create_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let mut digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
            digest.add_buffer(dataset);
            digest
        }
    };

    let create_sym_rcsketch = |accuracy_param: T| {
        move |dataset: &[T]| {
            let rc_digest = RCSketch::new(dataset.len(), accuracy_param.to_usize().unwrap());
            let mut digest = SymDigest::new(rc_digest.clone(), rc_digest.clone());
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
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let create_t_digest_asym = |compression_param: T| {
        move |dataset: &[T]| {
            let mut digest = TDigest::new(
                &scale_functions::k2_asym,
                &scale_functions::inv_k2_asym,
                compression_param,
            );
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let rcsketch_param = T::from(20.0).unwrap();
    let t_digest_param = T::from(1000.0).unwrap();

    let quantiles = [
        (1e-5, "1e-5"),
        (1e-4, "1e-4"),
        (1e-3, "1e-3"),
        (1e-2, "1e-2"),
        (1e-1, "1e-1"),
        (0.5, "0.5"),
        (1.0 - 1e-1, "0.9"),
        (1.0 - 1e-2, "0.99"),
        (1.0 - 1e-3, "0.999"),
        (1.0 - 1e-4, "0.9999"),
        (1.0 - 1e-5, "0.99999"),
    ]
    .iter()
    .map(|(q, marker)| (T::from(*q).unwrap(), marker.to_string()))
    .collect::<Vec<(T, String)>>();

    let input_size = 1_000_000;

    for (est_name, est_func, gen_est_position) in &get_estimation_funcs() {
        for (dist_name, dataset_func) in &get_distributions() {
            let digestfn = create_t_digest(t_digest_param);
            let mut x = digestfn(&dataset_func(input_size));
            for (quantile, quantile_string) in &quantiles {
                println!(
                    "{} at q={}: {:?}",
                    dist_name,
                    quantile_string,
                    x.est_value_at_quantile(*quantile)
                );
            }
            let mut series = Vec::new();
            let mut s = Vec::new();
            for (quantile, quantile_string) in &quantiles {
                let est_value = gen_est_position(*quantile, &dataset_func(input_size));
                let accuracy_measurements = sample_digest_accuracy(
                    create_rcsketch(rcsketch_param),
                    || dataset_func(input_size),
                    est_func(est_value),
                    error_func,
                    100,
                )
                .unwrap();
                s.push((T::from(*quantile).unwrap(), accuracy_measurements));
            }
            series.push(Line {
                name: format!("RCSketch, ap = {:?}", rcsketch_param),
                datapoints: s,
                colour: &RED,
                marker: None,
            });

            let mut s = Vec::new();
            for (quantile, quantile_string) in &quantiles {
                let est_value = gen_est_position(*quantile, &dataset_func(input_size));
                let accuracy_measurements = sample_digest_accuracy(
                    create_sym_rcsketch(rcsketch_param),
                    || dataset_func(input_size),
                    est_func(est_value),
                    error_func,
                    100,
                )
                .unwrap();
                s.push((T::from(*quantile).unwrap(), accuracy_measurements));
            }
            series.push(Line {
                name: format!("Sym RCSketch, ap = {:?}", rcsketch_param),
                datapoints: s,
                colour: &MAGENTA,
                marker: None,
            });

            let mut s = Vec::new();
            for (quantile, quantile_string) in &quantiles {
                let est_value = gen_est_position(*quantile, &dataset_func(input_size));
                let accuracy_measurements = sample_digest_accuracy(
                    create_t_digest(t_digest_param),
                    || dataset_func(input_size),
                    est_func(est_value),
                    error_func,
                    100,
                )
                .unwrap();
                s.push((T::from(*quantile).unwrap(), accuracy_measurements));
            }
            series.push(Line {
                name: format!("T-Digest, ap = {:?}", t_digest_param),
                datapoints: s,
                colour: &BLUE,
                marker: None,
            });

            let mut s = Vec::new();
            for (quantile, quantile_string) in &quantiles {
                let est_value = gen_est_position(*quantile, &dataset_func(input_size));
                let accuracy_measurements = sample_digest_accuracy(
                    create_t_digest_asym(t_digest_param),
                    || dataset_func(input_size),
                    est_func(est_value),
                    error_func,
                    100,
                )
                .unwrap();
                s.push((T::from(*quantile).unwrap(), accuracy_measurements));
            }
            series.push(Line {
                name: format!("T-Digest k2_asym, ap = {:?}", t_digest_param),
                datapoints: s,
                colour: &CYAN,
                marker: None,
            });

            plot_box_plot_graph(
                &format!(
                    "Relative error against quantile for estimate {}. Digest similarity. Dist: {}",
                    est_name, dist_name
                ),
                series,
                &Path::new(&format!(
                    "plots/err_vs_quantile_similarity_for_{}_{}.png",
                    est_name.to_lowercase().replace(" ", "_"),
                    dist_name.to_lowercase().replace(" ", "_")
                )),
                "Quantile",
                "Relative Error (ppm)",
            )
            .unwrap();
        }
    }
}

fn plot_error_against_input_size<T>()
where
    T: Float + Send + Sync + NumAssignOps + std::fmt::Debug,
{
    let test_func =
        |value: T| move |digest: &mut dyn Digest<T>| digest.est_value_at_quantile(value);

    let test_func_f64 =
        |value: f64| move |digest: &mut dyn Digest<f64>| digest.est_value_at_quantile(value);

    let error_func = |a, b| {
        T::max(
            T::from(1.0).unwrap(),
            relative_error(a, b) * T::from(1e6).unwrap(),
        )
    };

    let error_func_f64 = |a, b| f64::max(1.0, relative_error(a, b) * 1e6);

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
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let create_t_digest_f64 = |compression_param: f64| {
        move |dataset: &[f64]| {
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

    let create_t_digest_k2n = |compression_param: T| {
        move |dataset: &[T]| {
            let mut digest = TDigest::new(
                &scale_functions::k2n,
                &scale_functions::inv_k2n,
                compression_param,
            );
            dataset
                .chunks(T_DIGEST_CHUNK_SIZE)
                .for_each(|chunk| digest.add_buffer(chunk));
            digest
        }
    };

    let quantiles = [
        (1e-5, "X"),
        (1e-4, "▲"),
        (1e-3, "◆"),
        (1e-2, "●"),
        (1e-1, "■"),
    ]
    .iter()
    .map(|(q, marker)| (T::from(*q).unwrap(), marker.to_string()))
    .collect::<Vec<(T, String)>>();

    let rcsketch_param = T::from(20.0).unwrap();
    let t_digest_param = T::from(6000.0).unwrap();

    //    let input_sizes = [10_000, 100_000, 1_000_000];
    let input_sizes = [
        10_000,
        20_000,
        40_000,
        60_000,
        80_000,
        100_000,
        200_000,
        400_000,
        600_000,
        800_000,
        1_000_000,
        2_000_000,
        4_000_000,
        6_000_000,
        8_000_000,
        10_000_000,
        20_000_000,
        40_000_000,
        60_000_000,
        80_000_000,
        100_000_000,
        200_000_000,
        300_000_000,
        400_000_000,
        800_000_000,
        1_000_000_000,
    ];

    // let rc_sketch_mem_size = |input_size| {
    //     let digest = create_rcsketch(rcsketch_param)(&gen_uniform_tan_vec(input_size));
    //     digest.owned_size()
    // };
    // let t_digest_mem_size = |input_size| {
    //     let digest = create_t_digest(t_digest_param)(&gen_uniform_tan_vec(input_size));
    //     digest.owned_size()
    // };

    for (dist_name, dataset_func) in &get_distributions() {
        let mut series = Vec::new();
        for (quantile, marker) in &quantiles {
            let mut s = Vec::new();
            for input_size in &input_sizes {
                let accuracy_measurements = sample_digest_accuracy(
                    create_rcsketch(rcsketch_param),
                    || dataset_func(*input_size),
                    test_func(*quantile),
                    error_func,
                    10,
                )
                .unwrap();
                s.push((T::from(*input_size).unwrap(), accuracy_measurements));
            }

            series.push(
                Line {
                    name: format!("RCSketch, q = 1e{:?}", quantile.log10().to_i32().unwrap()),
                    datapoints: s,
                    colour: &RED,
                    marker: Some(marker.clone()),
                }
                .into_line_f64(),
            );

            let mut s = Vec::new();
            for input_size in &input_sizes {
                let accuracy_measurements = sample_digest_accuracy(
                    create_t_digest(t_digest_param),
                    || dataset_func(*input_size),
                    test_func(*quantile),
                    error_func,
                    10,
                )
                .unwrap();
                s.push((T::from(*input_size).unwrap(), accuracy_measurements));
            }

            series.push(
                Line {
                    name: format!("t-Digest, q = 1e{:?}", quantile.log10().to_i32().unwrap()),
                    datapoints: s,
                    colour: &BLUE,
                    marker: Some(marker.clone()),
                }
                .into_line_f64(),
            );

            let mut s = Vec::new();
            for input_size in &input_sizes {
                let accuracy_measurements = sample_digest_accuracy(
                    create_t_digest_f64(t_digest_param.to_f64().unwrap()),
                    || {
                        dataset_func(*input_size)
                            .into_iter()
                            .map(|x| x.to_f64().unwrap())
                            .collect()
                    },
                    test_func_f64(quantile.to_f64().unwrap()),
                    error_func_f64,
                    10,
                )
                .unwrap();
                s.push((*input_size as f64, accuracy_measurements));
            }

            series.push(
                Line {
                    name: format!(
                        "t-Digest f64, q = 1e{:?}",
                        quantile.log10().to_i32().unwrap()
                    ),
                    datapoints: s,
                    colour: &BLUE,
                    marker: Some(marker.clone()),
                }
                .into_line_f64(),
            );

            let mut s = Vec::new();
            for input_size in &input_sizes {
                let accuracy_measurements = sample_digest_accuracy(
                    create_t_digest_k2n(t_digest_param),
                    || dataset_func(*input_size),
                    test_func(*quantile),
                    error_func,
                    10,
                )
                .unwrap();
                s.push((T::from(*input_size).unwrap(), accuracy_measurements));
            }

            series.push(
                Line {
                    name: format!(
                        "t-Digest k2n, q = 1e{:?}",
                        quantile.log10().to_i32().unwrap()
                    ),
                    datapoints: s,
                    colour: &CYAN,
                    marker: Some(marker.clone()),
                }
                .into_line_f64(),
            );
        }
        plot_line_graph(
            &format!(
                "Error at against input size for value estimate at quantile. Dist: {}",
                dist_name
            ),
            series,
            &Path::new(&format!(
                "plots/err_vs_input_for_est_value_from_quantile_{}.png",
                dist_name.to_lowercase().replace(" ", "_")
            )),
            "Input size",
            "Relative Error (ppm)",
            false,
        )
        .unwrap();
    }
}

fn plot_memory_usage_against_compression_parameter<T>()
where
    T: Float + Sync + Send + NumAssignOps + std::fmt::Debug,
{
    let create_rcsketch = |dataset: &[T], param: T| {
        let mut digest = RCSketch::new(dataset.len(), param.to_usize().unwrap());
        digest.add_buffer(dataset);
        digest
    };

    let create_t_digest = |dataset: &[T], param: T| {
        let mut digest = TDigest::new(&scale_functions::k2, &scale_functions::inv_k2, param);
        dataset
            .chunks(T_DIGEST_CHUNK_SIZE)
            .for_each(|chunk| digest.add_buffer(chunk));
        digest
    };

    let mut series = Vec::new();
    let comp_params: Vec<u32> = (0..18).map(|x| 1 << x).collect();
    let mut s = Vec::new();
    for comp_param in &comp_params {
        let x = create_rcsketch(&gen_uniform_vec(10_000_000), T::from(*comp_param).unwrap());
        println!(
            "RC Digest n 10^7 with param {}: {} bytes",
            comp_param,
            x.owned_size()
        );
        s.push((
            T::from(*comp_param).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: "RCSketch, n = 10^7".to_string(),
        datapoints: s,
        colour: &RED,
        marker: None,
    });

    let mut s = Vec::new();
    for comp_param in &comp_params {
        let x = create_rcsketch(&gen_uniform_vec(100_000), T::from(*comp_param).unwrap());
        println!(
            "RC Digest n 10^5 with param {}: {} bytes",
            comp_param,
            x.owned_size()
        );
        s.push((
            T::from(*comp_param).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: "RCSketch, n = 10^5".to_string(),
        datapoints: s,
        colour: &RED,
        marker: None,
    });

    let mut s = Vec::new();
    for comp_param in &comp_params {
        let x = create_t_digest(&gen_uniform_vec(10_000_000), T::from(*comp_param).unwrap());
        println!(
            "T digest with param {}: {} bytes",
            comp_param,
            x.owned_size()
        );
        s.push((
            T::from(*comp_param).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: "T-Digest n = 10^7".to_string(),
        datapoints: s,
        colour: &BLUE,
        marker: None,
    });
    plot_line_graph(
        "Memory usage against compression/accuracy parameter",
        series,
        &Path::new("plots/mem_vs_comp_param.png"),
        "Compression/Accuracy parameter",
        "Memory usage bytes",
        false,
    )
    .unwrap();
}

fn plot_memory_usage_against_input_size<T>()
where
    T: Float + Sync + Send + NumAssignOps + std::fmt::Debug,
{
    let create_rcsketch = |dataset: &[T], param: T| {
        let mut digest = RCSketch::new(dataset.len(), param.to_usize().unwrap());
        digest.add_buffer(dataset);
        digest
    };

    let create_rcsketch2 = |dataset: &[T], param: T| {
        let mut digest = RCSketch2::new(param.to_usize().unwrap());
        digest.add_buffer(dataset);
        digest
    };

    let create_t_digest = |dataset: &[T], param: T| {
        let mut digest = TDigest::new(&scale_functions::k2, &scale_functions::inv_k2, param);
        dataset
            .chunks(T_DIGEST_CHUNK_SIZE)
            .for_each(|chunk| digest.add_buffer(chunk));
        digest
    };

    let input_sizes: Vec<i32> = (0..24).map(|x| 1 << x).collect();
    let rc_sketch_param = T::from(20.0).unwrap();
    let t_digest_param = T::from(6000.0).unwrap();

    let mut series = Vec::new();
    let mut s = Vec::new();
    for input_size in &input_sizes {
        let x = create_rcsketch(&gen_uniform_vec(*input_size), rc_sketch_param);
        println!(
            "RC Sketch n = {}, param = {}, size: {} bytes",
            input_size,
            rc_sketch_param.to_f64().unwrap(),
            x.owned_size()
        );
        s.push((
            T::from(*input_size).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: format!("RC Sketch param = {}", rc_sketch_param.to_f64().unwrap()),
        datapoints: s,
        colour: &RED,
        marker: None,
    });

    let mut s = Vec::new();
    for input_size in &input_sizes {
        let x = create_rcsketch2(&gen_uniform_vec(*input_size), rc_sketch_param);
        println!(
            "RC Sketch2 n = {}, param = {}, size: {} bytes",
            input_size,
            rc_sketch_param.to_f64().unwrap(),
            x.owned_size()
        );
        s.push((
            T::from(*input_size).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: format!("RC Sketch2 param = {}", rc_sketch_param.to_f64().unwrap()),
        datapoints: s,
        colour: &MAGENTA,
        marker: None,
    });

    let mut s = Vec::new();
    for input_size in &input_sizes {
        let x = create_t_digest(&gen_uniform_vec(*input_size), t_digest_param);
        println!(
            "T-Digest n = {}, param = {}, size: {} bytes",
            input_size,
            t_digest_param.to_f64().unwrap(),
            x.owned_size()
        );
        s.push((
            T::from(*input_size).unwrap(),
            vec![T::from(x.owned_size()).unwrap()],
        ));
    }

    series.push(Line {
        name: format!("T-Digest param = {}", t_digest_param.to_f64().unwrap()),
        datapoints: s,
        colour: &BLUE,
        marker: None,
    });
    plot_line_graph(
        "Memory usage against input_size parameter",
        series,
        &Path::new("plots/mem_vs_input_size.png"),
        "Input size",
        "Memory usage (bytes)",
        false,
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

fn relative_error<T>(measured: T, actual: T) -> T
where
    T: Float,
{
    // println!("{} {}", measured, actual);
    // if ((measured - actual).abs() / actual.abs() >= T::from(1.0).unwrap()) {
    //     println!(
    //         "Error {} {} {}",
    //         measured.to_f64().unwrap(),
    //         actual.to_f64().unwrap(),
    //         ((measured - actual).abs() / actual.abs()).to_f64().unwrap()
    //     );
    // }
    if actual == T::from(0.0).unwrap() {
        return measured.abs();
    }
    (measured - actual).abs() / actual.abs()
}

fn main() {
    // value_error_against_quantile::<f32>();
    // quantile_error_against_value::<f32>();
    // determine_required_parameter::<f32>();
    // determine_required_parameter::<f64>();
    plot_error_against_mem_usage::<f32>();
    // plot_error_against_mem_usage_parallel::<f32>();
    // plot_error_against_quantiles_full_range::<f32>();
    // plot_error_against_input_size::<f32>();
    // plot_memory_usage_against_compression_parameter::<f32>();
    plot_memory_usage_against_input_size::<f32>();
    println!("Complete");
}
