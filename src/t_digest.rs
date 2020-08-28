#![allow(dead_code)]
use std::ops::Add;

#[derive(Clone, Debug, PartialEq)]
pub struct Centroid {
    pub mean: f64,
    pub weight: f64,
}

impl Add for Centroid {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let new_weight = self.weight + other.weight;
        Centroid {
            mean: ((self.mean * self.weight) + (other.mean * other.weight)) / new_weight,
            weight: new_weight,
        }
    }
}

impl Add for &Centroid {
    type Output = Centroid;

    fn add(self, other: &Centroid) -> Centroid {
        let new_weight = self.weight + other.weight;
        Centroid {
            mean: ((self.mean * self.weight) + (other.mean * other.weight)) / new_weight,
            weight: new_weight,
        }
    }
}

pub struct TDigest<'a> {
    pub centroids: Vec<Centroid>,
    pub compress_factor: f64,
    pub scale_func: &'a dyn Fn(f64, f64) -> f64,
    pub inverse_scale_func: &'a dyn Fn(f64, f64) -> f64,
    pub min: f64,
    pub max: f64,
}

impl<'a> TDigest<'a> {
    pub fn new(
        scale_func: &'a dyn Fn(f64, f64) -> f64,
        inverse_scale_func: &'a dyn Fn(f64, f64) -> f64,
        compress_factor: f64,
    ) -> TDigest<'a> {
        TDigest {
            centroids: Vec::new(),
            compress_factor,
            scale_func,
            inverse_scale_func,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    pub fn add_buffer(&mut self, mut buffer: Vec<Centroid>) {
        self.update_limits(&buffer);
        buffer.extend(self.centroids.clone());
        buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        let num_elements: f64 = buffer.iter().map(|c| c.weight).sum();
        let mut w0 = 0.0;
        let get_w_limit = |w0| {
            (self.inverse_scale_func)(
                (self.scale_func)(w0 / num_elements, self.compress_factor) + 1.0,
                self.compress_factor,
            ) * num_elements
        };
        let mut w_limit = get_w_limit(w0);
        let mut new_centroids = Vec::new();

        let mut buffer_iter = buffer.into_iter();
        let mut current_centroid = buffer_iter.next().unwrap();
        for next_centroid in buffer_iter {
            let w = w0 + (current_centroid.weight + next_centroid.weight);

            if w <= w_limit {
                current_centroid = current_centroid + next_centroid;
            } else {
                w0 += current_centroid.weight;
                new_centroids.push(current_centroid);
                w_limit = get_w_limit(w0);
                current_centroid = next_centroid;
            }
        }
        new_centroids.push(current_centroid);
        self.centroids = new_centroids;
    }

    pub fn add_cluster(&mut self, clusters: Vec<Centroid>, growth_limit: f64) {
        self.update_limits(&clusters);
        for x in clusters {
            let min_dist = self
                .centroids
                .iter()
                .map(|c| (c.mean - x.mean).abs())
                .min_by(|a, b| a.partial_cmp(b).unwrap());
            match min_dist {
                Some(min) => {
                    let acceptable_centroids: Vec<bool> = self
                        .centroids
                        .iter()
                        .map(|c| {
                            (c.mean - x.mean).abs() == min && self.k_size(&(&x + c)).abs() < 1.0
                        })
                        .collect();

                    let acceptable_centroids: Vec<&mut Centroid> = self
                        .centroids
                        .iter_mut()
                        .zip(acceptable_centroids)
                        .filter(|(_c, ok)| *ok)
                        .map(|(c, _ok)| c)
                        .collect();

                    assert!(
                        acceptable_centroids.len() <= 2,
                        "acceptable centroids: {:?}, centroid: {:?}",
                        acceptable_centroids,
                        x
                    );
                    if acceptable_centroids.len() > 0 {
                        acceptable_centroids
                            .iter()
                            .min_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
                        let first = acceptable_centroids.into_iter().next().unwrap();
                        first.mean = (first.mean * first.weight + x.mean * x.weight)
                            / (first.weight + x.weight);
                        first.weight = first.weight + x.weight;
                    } else {
                        match self
                            .centroids
                            .binary_search_by(|probe| probe.mean.partial_cmp(&x.mean).unwrap())
                        {
                            Ok(index) => self.centroids.insert(index, x),
                            Err(index) => self.centroids.insert(index, x),
                        }
                    }
                }
                None => {
                    match self
                        .centroids
                        .binary_search_by(|probe| probe.mean.partial_cmp(&x.mean).unwrap())
                    {
                        Ok(index) => self.centroids.insert(index, x),
                        Err(index) => self.centroids.insert(index, x),
                    }
                }
            }

            if self.centroids.len() > (growth_limit * self.compress_factor) as usize {
                self.add_buffer(Vec::new());
            }
        }
        self.add_buffer(Vec::new());
    }

    pub fn weight_left(&self, target_centroid: &Centroid) -> f64 {
        self.centroids
            .iter()
            .filter(|c| c.mean < target_centroid.mean)
            .map(|c| c.weight)
            .sum()
    }

    pub fn total_weight(&self) -> f64 {
        self.centroids.iter().map(|c| c.weight).sum()
    }

    fn k_size(&self, target_centroid: &Centroid) -> f64 {
        let new_total_weight = self.total_weight() + target_centroid.weight;
        let q_left = self.weight_left(target_centroid) / new_total_weight;
        let q_right = q_left + target_centroid.weight / new_total_weight;
        (self.scale_func)(q_right, self.compress_factor)
            - (self.scale_func)(q_left, self.compress_factor)
    }

    pub fn interpolate(&self, quantile: f64) -> f64 {
        let total_count = self.total_weight();
        let mut current_quantile = 0.0;
        for i in 0..self.centroids.len() {
            // Quartile is located before this center of this centroid
            let new_quantile = current_quantile + (self.centroids[i].weight / (2.0 * total_count));
            if new_quantile > quantile {
                if i == 0 {
                    let prev_centroid = Centroid {
                        mean: self.min,
                        weight: 1.0,
                    };
                    return self.interpolate_centroids(
                        &prev_centroid,
                        &self.centroids[i],
                        quantile,
                        prev_centroid.weight / (2.0 * total_count),
                        total_count,
                    );
                } else {
                    let prev_centroid = &self.centroids[i - 1];
                    return self.interpolate_centroids(
                        prev_centroid,
                        &self.centroids[i],
                        quantile,
                        current_quantile,
                        total_count,
                    );
                }
            }
            current_quantile += self.centroids[i].weight / total_count;
        }
        let curr_centroid = Centroid {
            mean: self.max,
            weight: 1.0,
        };
        return self.interpolate_centroids(
            &self.centroids[self.centroids.len() - 1],
            &curr_centroid,
            quantile,
            current_quantile,
            total_count,
        );
    }

    fn interpolate_centroids(
        &self,
        prev_centroid: &Centroid,
        current_centroid: &Centroid,
        quantile: f64,
        current_quantile: f64,
        total_count: f64,
    ) -> f64 {
        let prev_quantile = current_quantile - (prev_centroid.weight / (2.0 * total_count));
        let quantile_proportion = (quantile - prev_quantile)
            / (current_quantile + (current_centroid.weight / (2.0 * total_count)) - prev_quantile);
        println!("{:?}", current_quantile);
        return quantile_proportion * (current_centroid.mean - prev_centroid.mean)
            + prev_centroid.mean;
    }

    fn update_limits(&mut self, centroids: &Vec<Centroid>) {
        self.min = centroids
            .iter()
            .map(|x| x.mean)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_else(|| self.min);

        self.max = centroids
            .iter()
            .map(|x| x.mean)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_else(|| self.max);
    }
}

pub mod scale_functions {
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
    use crate::t_digest::scale_functions::{inv_k0, inv_k1, k0, k1};
    use crate::t_digest::Centroid;
    use crate::t_digest::TDigest;
    use approx::assert_relative_eq;

    #[test]
    fn add_buffer_with_single_centroid() {
        let buffer = vec![Centroid {
            mean: 1.0,
            weight: 1.0,
        }];
        let mut digest = TDigest::new(&k0, &inv_k0, 1.0);
        digest.add_buffer(buffer);

        assert_eq!(digest.centroids.len(), 1);
        assert_eq!(digest.centroids[0].mean, 1.0);
        assert_eq!(digest.centroids[0].weight, 1.0);
        assert_eq!(digest.total_weight(), 1.0);
    }

    #[test]
    fn add_buffer_with_multiple_centroid() {
        let buffer = vec![
            Centroid {
                mean: 1.0,
                weight: 1.0,
            },
            Centroid {
                mean: 2.0,
                weight: 1.0,
            },
            Centroid {
                mean: 0.5,
                weight: 1.0,
            },
        ];
        let mut digest = TDigest::new(&k0, &inv_k0, 50.0);
        digest.add_buffer(buffer);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.interpolate(0.0), 0.5);
        assert_relative_eq!(digest.interpolate(0.25), 0.625);
        assert_relative_eq!(digest.interpolate(0.5), 1.0);
        assert_relative_eq!(digest.interpolate(0.75), 1.75);
        assert_relative_eq!(digest.interpolate(1.0), 2.0);
        assert_eq!(digest.total_weight(), 3.0);
    }

    #[test]
    fn add_buffer_with_many_centroids() {
        let buffer = (0..1001)
            .map(|x| Centroid {
                mean: x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 100.0);
        digest.add_buffer(buffer);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.interpolate(0.0), 0.0);
        assert_relative_eq!(digest.interpolate(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(0.5), 500.0, epsilon = 0.0000001);
        assert_relative_eq!(digest.interpolate(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }

    #[test]
    fn add_buffer_with_many_centroids_high_compression() {
        let buffer = (0..1001)
            .map(|x| Centroid {
                mean: x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_buffer(buffer);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.interpolate(0.0), 0.0);
        assert_relative_eq!(digest.interpolate(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(0.5), 500.0, epsilon = 0.0000001);
        assert_relative_eq!(digest.interpolate(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }

    #[test]
    fn add_cluster_with_single_centroid() {
        let cluster = vec![Centroid {
            mean: 1.0,
            weight: 1.0,
        }];
        let mut digest = TDigest::new(&k0, &inv_k0, 1.0);
        digest.add_cluster(cluster, 3.0);

        assert_eq!(digest.centroids.len(), 1);
        assert_eq!(digest.centroids[0].mean, 1.0);
        assert_eq!(digest.centroids[0].weight, 1.0);
        assert_eq!(digest.total_weight(), 1.0);
    }

    #[test]
    fn add_cluster_with_multiple_centroid() {
        let cluster = vec![
            Centroid {
                mean: 1.0,
                weight: 1.0,
            },
            Centroid {
                mean: 2.0,
                weight: 1.0,
            },
            Centroid {
                mean: 0.5,
                weight: 1.0,
            },
        ];
        let mut digest = TDigest::new(&k0, &inv_k0, 50.0);
        digest.add_cluster(cluster, 3.0);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.interpolate(0.0), 0.5);
        assert_relative_eq!(digest.interpolate(0.25), 0.625);
        assert_relative_eq!(digest.interpolate(0.5), 1.0);
        assert_relative_eq!(digest.interpolate(0.75), 1.75);
        assert_relative_eq!(digest.interpolate(1.0), 2.0);
        assert_eq!(digest.total_weight(), 3.0);
    }

    #[test]
    fn add_cluster_with_many_centroids() {
        let cluster = (0..1001)
            .map(|x| Centroid {
                mean: x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 100.0);
        digest.add_cluster(cluster, 3.0);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.interpolate(0.0), 0.0);
        assert_relative_eq!(digest.interpolate(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(0.5), 500.0, epsilon = 0.0000001);
        assert_relative_eq!(digest.interpolate(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }

    #[test]
    fn add_cluster_with_many_centroids_high_compression() {
        let cluster: Vec<Centroid> = (0..1001)
            .map(|x| Centroid {
                mean: x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 20.0);
        digest.add_cluster(cluster, 10.0);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.interpolate(0.0), 0.0);
        assert_relative_eq!(digest.interpolate(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(0.5), 500.0, epsilon = 0.0000001);
        assert_relative_eq!(digest.interpolate(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.interpolate(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }
}

#[cfg(test)]
mod scale_functions_test {
    use crate::t_digest::scale_functions::{inv_k0, inv_k1, inv_k2, inv_k3, k0, k1, k2, k3};
    use approx::assert_relative_eq;

    #[test]
    fn k0_properties() {
        assert_relative_eq!(k0(0.0, 10.0), 0.0);
    }

    #[test]
    fn inv_k0_properties() {
        for i in 0..100 {
            assert_relative_eq!(inv_k0(k0(i as f64, 10.0), 10.0), i as f64);
        }
    }

    #[test]
    fn k1_properties() {
        assert_relative_eq!(k1(1.0, 10.0), 10.0 / 4.0);
    }

    #[test]
    fn inv_k1_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k1(k1(q, 10.0), 10.0), q);
        }
    }

    #[test]
    fn inv_k2_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k2(k2(q, 10.0), 10.0), q);
        }
    }

    #[test]
    fn inv_k3_properties() {
        for i in 1..99 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k3(k3(q, 10.0), 10.0), q, epsilon = 0.01);
        }
    }
}
