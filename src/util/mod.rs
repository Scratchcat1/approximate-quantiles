use crate::t_digest::centroid::Centroid;
use crate::traits::Digest;
use crate::util::linear_digest::LinearDigest;
use num_traits::Float;
// use rand::distributions::Uniform;
use rand_distr::{Distribution, Exp, Uniform};
use rayon::prelude::*;
pub mod keyed_sum_tree;
pub mod linear_digest;

pub fn weighted_average<F>(x1: F, w1: F, x2: F, w2: F) -> F
where
    F: Float,
{
    let weighted = (x1 * w1 + x2 * w2) / (w1 + w2);
    let max = F::max(x1, x2);
    let min = F::min(x1, x2);
    F::max(min, F::min(weighted, max))
}

/// Generate a vector of values from a uniform distribution
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_uniform_vec<F>(size: i32) -> Vec<F>
where
    F: Float,
{
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(0.0..1001.0);
    return (0..size)
        .map(|_| F::from(uniform.sample(&mut rng)).unwrap())
        .collect();
}

/// Generate a vector of values from a uniform distribution applied to the tan function
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_uniform_tan_vec<F>(size: i32) -> Vec<F>
where
    F: Float,
{
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(-2.0..2.0);
    return (0..size)
        .map(|_| F::from(uniform.sample(&mut rng).tan()).unwrap())
        .collect();
}

/// Generate a vector of values from a exp distribution
/// # Arguments
/// `size` Size of the vector to generate
/// `lambda` Lambda to pass to the exponential distribution
pub fn gen_uniform_exp_vec<F>(size: i32, lambda: F) -> Vec<F>
where
    F: Float,
{
    let mut rng = rand::thread_rng();
    let exp = Exp::new(lambda.to_f64().unwrap()).unwrap();
    return (0..size)
        .map(|_| F::from(-exp.sample(&mut rng)).unwrap())
        .collect();
}

/// Generate a vector of values from a reversed growing blocks distribution
/// See: Accuracy plots > input at https://datasketches.apache.org/docs/Quantiles/KllSketchVsTDigest.html
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_reverse_growing_blocks_vec<F>(size: i32) -> Vec<F>
where
    F: Float,
{
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(0.0..1.0);
    let mut increment_prob = 0.001;
    let mut val = F::from(1.0).unwrap();
    let mut elements = Vec::new();
    for _ in 0..size {
        if uniform.sample(&mut rng) <= increment_prob {
            increment_prob = increment_prob * 0.98;
            val = val - F::from(1.0).unwrap();
        }
        elements.push(val);
    }
    elements
}

/// Generate a vector of 1-weighted centroids from a uniform distribution
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_uniform_centroid_vec<F>(size: i32) -> Vec<Centroid<F>>
where
    F: Float,
{
    return gen_uniform_vec(size)
        .into_iter()
        .map(|x| Centroid {
            mean: x,
            weight: F::from(1.0).unwrap(),
        })
        .collect();
}

/// Generate a vector of random-weighted centroids from a uniform distribution
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_uniform_centroid_random_weight_vec<F>(size: i32) -> Vec<Centroid<F>>
where
    F: Float,
{
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(0.0..1001.0);
    return (0..size)
        .map(|_| Centroid {
            mean: F::from(uniform.sample(&mut rng)).unwrap(),
            weight: F::from(uniform.sample(&mut rng)).unwrap(),
        })
        .collect();
}

/// Generate a vector of ascending values 0, 1, .., (size - 1)
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_asc_vec<F>(size: i32) -> Vec<F>
where
    F: Float,
{
    return (0..size).map(|x| F::from(x).unwrap()).collect();
}

/// Generate a vector of ascending 1-weighed centroids with means 0, 1, .., (size - 1)
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_asc_centroid_vec<F>(size: i32) -> Vec<Centroid<F>>
where
    F: Float,
{
    return (0..size)
        .map(|x| Centroid {
            mean: F::from(x).unwrap(),
            weight: F::from(1.0).unwrap(),
        })
        .collect();
}

/// Narrow down the minimum required compression/accuracy parameter for a digest to pass a test function
/// # Arguments
/// `create_digest` Function which takes a dataset and compression/accuracy parameter and returns a filled digest
/// `gen_dataset` Function which generates a dataset for the digest
/// `test_func` Function which takes the filled digest and a linear digest to query the true values and returns success or failure if accuracy requirements are met
/// `test_count` Number of tests to perform each iteration
/// `pass_ratio` The proportion of tests which have to pass to consider the iteration a pass
/// `max_param` The max value of the parameter which will be considered
/// `epsilon` Minimum difference between the current value and optimal value at which to accept
pub fn opt_accuracy_parameter<D, C, G, T, F>(
    create_digest: C,
    gen_dataset: G,
    test_func: T,
    test_count: u32,
    pass_ratio: f64,
    max_param: F,
    epsilon: F,
) -> Result<F, String>
where
    C: Fn(&[F], F) -> D + Sync + Send,
    G: Fn() -> Vec<F> + Sync + Send,
    T: Fn(&mut dyn Digest<F>, &mut LinearDigest<F>) -> bool + Sync + Send,
    D: Digest<F>,
    F: Float + Sync + Send,
{
    let mut high = max_param;
    let mut low = F::from(0.0).unwrap();
    let mut current_param = (high + low) / F::from(2.0).unwrap();
    loop {
        let pass_count = (0..test_count)
            .into_par_iter()
            .filter(|_| {
                let dataset = gen_dataset();
                let mut linear_digest = LinearDigest::new();
                linear_digest.add_buffer(&dataset);
                let mut digest = create_digest(&dataset, current_param);
                test_func(&mut digest, &mut linear_digest)
            })
            .count();

        println!(
            "pass {}, low: {:?}, current: {:?}, high: {:?}",
            pass_count,
            low.to_f64().unwrap(),
            current_param.to_f64().unwrap(),
            high.to_f64().unwrap()
        );

        if (pass_count as f64 / test_count as f64) >= pass_ratio {
            high = current_param;
        } else {
            low = current_param;
        }
        if high - low < epsilon {
            // Return high as high is always guaranteed to have passed the test.
            return Ok(high);
        }
        current_param = (high + low) / F::from(2.0).unwrap();
    }
}

/// Sample the error for a digest and test function compared to the generated datasets.
/// # Arguments
/// `create_digest` Function which takes a dataset and compression/accuracy parameter and returns a filled digest
/// `gen_dataset` Function which generates a dataset for the digest
/// `test_func` Function which takes a digest and performs the desired query, returning that value.
/// `error_func` Function which takes the measured and actual errors as parameters and returns the error.
/// `test_count` Number of tests to perform each iteration
/// # Returns
/// `return` Vector of errors compared to the generated datasets
pub fn sample_digest_accuracy<D, C, G, T, E, F>(
    create_digest: C,
    gen_dataset: G,
    test_func: T,
    error_func: E,
    test_count: u32,
) -> Result<Vec<F>, String>
where
    C: Fn(&[F]) -> D + Send + Sync,
    G: Fn() -> Vec<F> + Send + Sync,
    T: Fn(&mut dyn Digest<F>) -> F + Send + Sync,
    D: Digest<F>,
    E: Fn(F, F) -> F + Send + Sync,
    F: Float + Send + Sync,
{
    let results: Vec<F> = (0..test_count)
        .into_par_iter()
        .map(|_| {
            let dataset = gen_dataset();
            let mut linear_digest = LinearDigest::new();
            linear_digest.add_buffer(&dataset);
            let mut digest = create_digest(&dataset);
            error_func(test_func(&mut digest), test_func(&mut linear_digest))
        })
        .collect();
    Ok(results)
}

// #[cfg(test)]
// mod test {
//     use approx::assert_relative_eq;
//     use crate::util::gen_asc_vec;

//     fn test_opt_accuracy_parameter_basic() {
//         assert_relative_eq!();
//     }
// }
