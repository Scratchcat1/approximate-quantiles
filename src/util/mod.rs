use crate::t_digest::centroid::Centroid;
use crate::traits::Digest;
use crate::util::linear_digest::LinearDigest;
use rand::distributions::{Distribution, Uniform};

pub mod keyed_sum_tree;
pub mod linear_digest;

pub fn weighted_average(x1: f64, w1: f64, x2: f64, w2: f64) -> f64 {
    let weighted = (x1 * w1 + x2 * w2) / (w1 + w2);
    let max = f64::max(x1, x2);
    let min = f64::min(x1, x2);
    f64::max(min, f64::min(weighted, max))
}

/// Generate a vector of values from a uniform distribution
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_uniform_vec(size: i32) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(0.0..1001.0);
    return (0..size).map(|_| uniform.sample(&mut rng) as f64).collect();
}

/// Generate a vector of 1-weighted centroids from a uniform distribution
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_uniform_centroid_vec(size: i32) -> Vec<Centroid> {
    return gen_uniform_vec(size)
        .into_iter()
        .map(|x| Centroid {
            mean: x,
            weight: 1.0,
        })
        .collect();
}

/// Generate a vector of random-weighted centroids from a uniform distribution
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_uniform_centroid_random_weight_vec(size: i32) -> Vec<Centroid> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::from(0.0..1001.0);
    return (0..size)
        .map(|_| Centroid {
            mean: uniform.sample(&mut rng) as f64,
            weight: uniform.sample(&mut rng) as f64,
        })
        .collect();
}

/// Generate a vector of ascending values 0, 1, .., (size - 1)
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_asc_vec(size: i32) -> Vec<f64> {
    return (0..size).map(|x| x as f64).collect();
}

/// Generate a vector of ascending 1-weighed centroids with means 0, 1, .., (size - 1)
/// # Arguments
/// `size` Size of the vector to generate
pub fn gen_asc_centroid_vec(size: i32) -> Vec<Centroid> {
    return (0..size)
        .map(|x| Centroid {
            mean: x as f64,
            weight: 1.0,
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
pub fn opt_accuracy_parameter<D, C, G, T>(
    create_digest: C,
    gen_dataset: G,
    test_func: T,
    test_count: u32,
    pass_ratio: f64,
    max_param: f64,
    epsilon: f64,
) -> Result<f64, String>
where
    C: Fn(&[f64], f64) -> D,
    G: Fn() -> Vec<f64>,
    T: Fn(&mut dyn Digest, &mut LinearDigest) -> bool,
    D: Digest,
{
    let mut high = max_param;
    let mut low = 0.0;
    let mut current_param = (high + low) / 2.0;
    loop {
        let mut pass_count = 0;
        for _ in 0..test_count {
            let dataset = gen_dataset();
            let mut linear_digest = LinearDigest::new();
            linear_digest.add_buffer(&dataset);
            let mut digest = create_digest(&dataset, current_param);
            if test_func(&mut digest, &mut linear_digest) {
                pass_count += 1;
            }
        }

        println!(
            "pass {}, low: {}, current: {}, high: {}",
            pass_count, low, current_param, high
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
        current_param = (high + low) / 2.0;
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
pub fn sample_digest_accuracy<D, C, G, T, E>(
    create_digest: C,
    gen_dataset: G,
    test_func: T,
    error_func: E,
    test_count: u32,
) -> Result<Vec<f64>, String>
where
    C: Fn(&[f64]) -> D,
    G: Fn() -> Vec<f64>,
    T: Fn(&mut dyn Digest) -> f64,
    D: Digest,
    E: Fn(f64, f64) -> f64,
{
    let mut results = Vec::new();
    for _ in 0..test_count {
        let dataset = gen_dataset();
        let mut linear_digest = LinearDigest::new();
        linear_digest.add_buffer(&dataset);
        let mut digest = create_digest(&dataset);
        results.push(error_func(
            test_func(&mut digest),
            test_func(&mut linear_digest),
        ));
    }
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
