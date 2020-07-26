use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Centroid {
    pub mean: f64,
    pub weight: f64,
}

impl Add for Centroid {
    type Output = Centroid;

    fn add(self, other: Centroid) -> Centroid {
        let new_weight = self.weight + other.weight;
        Centroid {
            mean: ((self.mean * self.weight) + (other.mean * other.weight)) / new_weight,
            weight: new_weight,
        }
    }
}

struct TDigest<'a> {
    pub centroids: Vec<Centroid>,
    pub compress_factor: f64,
    pub scale_func: &'a dyn Fn(f64, f64) -> f64,
    pub inverse_scale_func: &'a dyn Fn(f64, f64) -> f64,
}

impl<'a> TDigest<'a> {
    pub fn new(
        scale_func: &'a dyn Fn(f64, f64) -> f64,
        inverse_scale_func: &'a dyn Fn(f64, f64) -> f64,
    ) -> TDigest<'a> {
        TDigest {
            centroids: Vec::new(),
            compress_factor: 1.0,
            scale_func,
            inverse_scale_func,
        }
    }
    pub fn add_buffer(&mut self, buffer: Vec<Centroid>) {
        let mut sorted = [self.centroids.clone(), buffer].concat();
        sorted.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());

        let num_elements: f64 = sorted.iter().map(|c| c.weight).sum();
        let mut q0 = 0.0;
        let q_limit = (self.inverse_scale_func)(
            (self.scale_func)(q0, self.compress_factor),
            self.compress_factor,
        );
        let mut new_centroids = Vec::new();

        let mut current_centroid = sorted[0].clone();
        for i in 2..(sorted.len()) {
            let next_centroid = sorted[i].clone();
            let q = q0 + (current_centroid.weight + next_centroid.weight) / num_elements;

            if q <= q_limit {
                current_centroid = current_centroid + next_centroid;
            } else {
                q0 = q0 + current_centroid.weight / num_elements;
                new_centroids.push(current_centroid);
                current_centroid = next_centroid;
            }
        }
        new_centroids.push(current_centroid);
        self.centroids = new_centroids;
    }

    pub fn add_cluster(&mut self, clusters: Vec<Centroid>) {
        for x in clusters {
            let min_dist = self
                .centroids
                .iter()
                .map(|c| c.mean - x.mean)
                .min_by(|a, b| a.partial_cmp(b).unwrap());
            match min_dist {
                Some(min) => {
                    let mut acceptable_centroids: Vec<&mut Centroid> = self
                        .centroids
                        .iter_mut()
                        .filter(|c| (c.mean - x.mean) == min && (c.weight + x.weight) < 1.0)
                        .collect();

                    if acceptable_centroids.len() > 0 {
                        acceptable_centroids.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
                        let first = acceptable_centroids.first().unwrap();
                        // first.weight = first.weight + x.weight;
                    }
                }
                None => {}
            }
        }
    }
}

mod scale_functions {
    use std::f64::consts::PI;
    pub fn k0(quartile: f64, comp_factor: f64) -> f64 {
        (quartile * comp_factor) / 2.0
    }

    pub fn inv_k0(scale: f64, comp_factor: f64) -> f64 {
        (scale * 2.0) / comp_factor
    }

    pub fn k1(quartile: f64, comp_factor: f64) -> f64 {
        (comp_factor / (2.0 * PI)) * (2.0 * quartile - 1.0).asin()
    }

    pub fn inv_k1(scale: f64, comp_factor: f64) -> f64 {
        (1.0 + (2.0 * PI * scale / comp_factor).sin()) / 2.0
    }

    pub fn k2(quartile: f64, comp_factor: f64) -> f64 {
        let n = 10.0;
        (comp_factor / (4.0 * (n / comp_factor).log10() + 24.0))
            * (quartile / (1.0 - quartile)).log10()
    }

    pub fn inv_k2(scale: f64, comp_factor: f64) -> f64 {
        let n: f64 = 10.0;
        let x =
            (10.0 as f64).powf((scale * (4.0 * (n / comp_factor).log10() + 24.0)) / comp_factor);
        return x / (1.0 + x);
    }

    pub fn k3(quartile: f64, comp_factor: f64) -> f64 {
        let n = 10.0;
        let factor = match quartile <= 0.5 {
            true => (2.0 * quartile).log10(),
            false => -(2.0 * (1.0 - quartile)).log10(),
        };
        (comp_factor / (4.0 * (n / comp_factor).log10() + 21.0)) * factor
    }

    pub fn inv_k3(scale: f64, comp_factor: f64) -> f64 {
        let n = 10.0;
        let pow = (scale * (4.0 * (n / comp_factor).log10() + 21.0)) / comp_factor;

        let q_low = (10.0 as f64).powf(pow) / 2.0;
        let q_high = (2.0 - (10.0 as f64).powf(-pow)) / 2.0;
        match (0.5 - q_low).abs() > (0.5 - q_high).abs() {
            true => q_high,
            false => q_low,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::t_digest::Centroid;
    use crate::t_digest::TDigest;

    #[test]
    fn add_buffer_with_single_centroid() {
        let buffer = vec![Centroid {
            mean: 1.0,
            weight: 1.0,
        }];
        let scale_func = |a, b| a;
        let inverse_scale_func = |a, b| a;
        let mut digest = TDigest::new(&scale_func, &inverse_scale_func);
        digest.add_buffer(buffer);

        println!("{:?}", digest.centroids);

        assert_eq!(digest.centroids.len(), 1);
        assert_eq!(digest.centroids[0].mean, 1.0);
        assert_eq!(digest.centroids[0].weight, 1.0);
    }
}

#[cfg(test)]
mod scale_functions_test {
    use crate::t_digest::scale_functions::{inv_k0, inv_k1, inv_k2, inv_k3, k0, k1, k2, k3};

    #[test]
    fn k0_properties() {
        approx::assert_relative_eq!(k0(0.0, 10.0), 0.0);
    }

    #[test]
    fn inv_k0_properties() {
        for i in 0..100 {
            approx::assert_relative_eq!(inv_k0(k0(i as f64, 10.0), 10.0), i as f64);
        }
    }

    #[test]
    fn k1_properties() {
        approx::assert_relative_eq!(k1(1.0, 10.0), 10.0 / 4.0);
    }

    #[test]
    fn inv_k1_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            approx::assert_relative_eq!(inv_k1(k1(q, 10.0), 10.0), q);
        }
    }

    #[test]
    fn inv_k2_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            approx::assert_relative_eq!(inv_k2(k2(q, 10.0), 10.0), q);
        }
    }

    #[test]
    fn inv_k3_properties() {
        for i in 1..99 {
            let q = i as f64 / 100.0;
            approx::assert_relative_eq!(inv_k3(k3(q, 10.0), 10.0), q, epsilon = 0.01);
        }
    }
}
