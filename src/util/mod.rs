use crate::t_digest::centroid::Centroid;
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
