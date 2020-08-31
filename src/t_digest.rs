#![allow(dead_code)]
use std::ops::Add;

/// Weighted value used in `TDigest`
#[derive(Clone, Debug, PartialEq)]
pub struct Centroid {
    /// Mean of the centroid
    pub mean: f64,
    /// Weight of the centroid
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
    /// Vector of centroids
    pub centroids: Vec<Centroid>,
    /// Compression factor to adjust the number of centroids to keep
    pub compress_factor: f64,
    /// Scale function to map a quantile to a unit-less value to limit the size of a centroid
    pub scale_func: &'a dyn Fn(f64, f64) -> f64,
    /// Function to invert the scale function
    pub inverse_scale_func: &'a dyn Fn(f64, f64) -> f64,
    /// Keeps track of the maximum value observed
    pub min: f64,
    /// Keeps track of the minimum value observed
    pub max: f64,
}

impl<'a> TDigest<'a> {
    /// Returns a new `TDigest`
    /// # Arguments
    ///
    /// * `scale_func` Scale function
    /// * `inverse_scale_func` Inverse scale function
    /// * `compress_factor` Compression factor
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

    /// Merge a buffer into the digest
    ///
    /// # Arguments
    ///
    /// * `buffer` The buffer to merge into the digest
    pub fn add_buffer(&mut self, mut buffer: Vec<Centroid>) {
        self.update_limits(&buffer);

        // Merge the digest centroids into the buffer since normally |buffer| > |self.centroids|
        buffer.extend(self.centroids.clone());
        // Nothing to merge, exit
        if buffer.len() == 0 {
            return;
        }
        buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        let num_elements: f64 = buffer.iter().map(|c| c.weight).sum();

        // Use weights instead of quantiles to avoid division in the main loop
        let mut w0 = 0.0;
        let get_w_limit = |w0| {
            (self.inverse_scale_func)(
                (self.scale_func)(w0 / num_elements, self.compress_factor) + 1.0,
                self.compress_factor,
            ) * num_elements
        };
        let mut w_size_limit = get_w_limit(w0);
        let mut new_centroids = Vec::new();

        let mut buffer_iter = buffer.into_iter();
        let first_centroid = buffer_iter.next().unwrap();

        // Keep track of the centroids which can be merged together instead of adding together each loop.
        // This minimises the number of calculations in the hot loop and may allow the merging to be vectorised.
        // The centroids are merged as soon as their combined weight would exceed the limit
        let mut mergeable_weight = first_centroid.weight;
        let mut mergeable_centroids = vec![first_centroid];
        for next_centroid in buffer_iter {
            let new_w = mergeable_weight + next_centroid.weight;

            // If combined weight is below the limit merge the centroids
            if new_w <= w_size_limit {
                mergeable_weight = new_w;
                mergeable_centroids.push(next_centroid);
            } else {
                // Combined weight exceeds limit, add the current centroid to the vector and calculate the new limit
                w0 += mergeable_weight;
                new_centroids.push(Self::merge_centroids(
                    mergeable_weight,
                    &mergeable_centroids,
                ));
                w_size_limit = get_w_limit(w0) - w0;
                mergeable_centroids.clear();
                mergeable_weight = next_centroid.weight;
                mergeable_centroids.push(next_centroid);
            }
        }
        new_centroids.push(Self::merge_centroids(
            mergeable_weight,
            &mergeable_centroids,
        ));
        self.centroids = new_centroids;
    }

    /// Merge a buffer of centroids into a single one
    /// # Arguments
    /// * `weight` Total weight of the centroids
    /// * `buffer` Centroids to merge
    fn merge_centroids(weight: f64, buffer: &[Centroid]) -> Centroid {
        let new_sum: f64 = buffer.iter().map(|c| c.mean * c.weight).sum();
        Centroid {
            mean: new_sum / weight,
            weight: weight,
        }
    }

    /// Add centroids to the digest via clustering
    ///
    /// # Arguments
    /// * `clusters` Centroids to add to the digest
    /// * `growth_limit` Factor to limit excessive growth of the digest by merging periodically
    pub fn add_cluster(&mut self, clusters: Vec<Centroid>, growth_limit: f64) {
        self.update_limits(&clusters);
        for x in clusters {
            let close_centroids = self.find_closest_centroids(&x);
            match close_centroids {
                Some(indexes) => {
                    // let acceptable_centroids_ok: Vec<bool> = indexes
                    //     .clone()
                    //     .map(|i| self.k_size(&(&x + &self.centroids[i])).abs() < 1.0)
                    //     .collect();

                    // let acceptable_centroids: Vec<&mut Centroid> = self.centroids[indexes]
                    //     .iter_mut()
                    //     .zip(acceptable_centroids_ok)
                    //     .filter(|(_c, ok)| *ok)
                    //     .map(|(c, _ok)| c)
                    //     .collect();

                    // Find the index of a centroid with space to merge the current centroid
                    // selecting the one with the minimum weight
                    let mut min_acceptable_index = None;
                    for index in indexes {
                        if self.k_size(&(&x + &self.centroids[index])).abs() < 1.0 {
                            match min_acceptable_index {
                                None => min_acceptable_index = Some(index),
                                Some(other_index) => {
                                    if self.centroids[other_index].mean
                                        * self.centroids[other_index].weight
                                        > self.centroids[index].mean * self.centroids[index].weight
                                    {
                                        min_acceptable_index = Some(index)
                                    }
                                }
                            }
                        }
                    }

                    match min_acceptable_index {
                        Some(index) => {
                            // Merge the current centroid with the centroid in the digest
                            let mut merge = &mut self.centroids[index];
                            merge.mean = (merge.mean * merge.weight + x.mean * x.weight)
                                / (merge.weight + x.weight);
                            merge.weight += x.weight;
                        }
                        None => {
                            // No suitable centroid in the digest was found, insert the current centroid into the digest
                            match self
                                .centroids
                                .binary_search_by(|probe| probe.mean.partial_cmp(&x.mean).unwrap())
                            {
                                Ok(index) => self.centroids.insert(index, x),
                                Err(index) => self.centroids.insert(index, x),
                            }
                        }
                    }
                }
                None => {
                    // No suitable centroid in the digest was found, insert the current centroid into the digest
                    match self
                        .centroids
                        .binary_search_by(|probe| probe.mean.partial_cmp(&x.mean).unwrap())
                    {
                        Ok(index) => self.centroids.insert(index, x),
                        Err(index) => self.centroids.insert(index, x),
                    }
                }
            }

            // Prevent excess growth with particular insertion patterns by periodically merging
            if self.centroids.len() > (growth_limit * self.compress_factor) as usize {
                self.add_buffer(Vec::new());
            }
        }
        self.add_buffer(Vec::new());
    }

    /// Find the range of indexes in the digest which cover the centroids which all have the minimum distance
    /// to the mean of the target centroid
    /// # Arguments
    ///
    /// * `target` Centroid to compare the mean of
    pub fn find_closest_centroids(&self, target: &Centroid) -> Option<std::ops::Range<usize>> {
        if self.centroids.len() == 0 {
            // No centroids are present so there are no closest centroids
            return None;
        }

        // Find the index of centroids closest to the target centroid
        let index = match self
            .centroids
            .binary_search_by(|probe| probe.mean.partial_cmp(&target.mean).unwrap())
        {
            Ok(index) => index,
            Err(index) => index,
        };
        let min_lr_dist;
        let mut left_index = index;
        let mut right_index = index + 1;
        if index == 0 {
            // Target is smallest centroid, min dist is to the right
            min_lr_dist = self.centroids[index].mean - target.mean;
        } else if index == self.centroids.len() {
            // Target is largest centroid, min dist is to the left
            min_lr_dist = self.centroids[index - 1].mean - target.mean;
            // Closest centroid is the last one
            left_index = index - 1;
            right_index = index;
        } else {
            // Determine if the minimum dist is to the left or the right
            let lower_diff = self.centroids[index - 1].mean - target.mean;
            let higher_diff = self.centroids[index].mean - target.mean;
            min_lr_dist = if lower_diff <= higher_diff {
                lower_diff
            } else {
                higher_diff
            };
        }

        // Handle the case where there are multiple centroids with the same mean by shifting the indexes
        // to include all with the minimum distance to the target
        while left_index > 0 && self.centroids[left_index - 1].mean - target.mean == min_lr_dist {
            left_index -= 1;
        }
        while right_index < self.centroids.len() - 1
            && self.centroids[right_index + 1].mean - target.mean == min_lr_dist
        {
            right_index += 1;
        }
        Some(left_index..right_index)
    }

    /// Calculate the weight of all centroids in the digest which have a lower mean than the target
    /// # Arguments
    ///
    /// * `target_centroid` Centroid to compare to
    pub fn weight_left(&self, target_centroid: &Centroid) -> f64 {
        self.centroids
            .iter()
            .filter(|c| c.mean < target_centroid.mean)
            .map(|c| c.weight)
            .sum()
    }

    /// Get the total weight of the digest
    pub fn total_weight(&self) -> f64 {
        self.centroids.iter().map(|c| c.weight).sum()
    }

    /// Calculate the k_size for the target centroid
    /// This is the scaled different between the left and right quartile of the centroid
    pub fn k_size(&self, target_centroid: &Centroid) -> f64 {
        let new_total_weight = self.total_weight() + target_centroid.weight;

        // Calculate the left and right quartiles
        let q_left = self.weight_left(target_centroid) / new_total_weight;
        let q_right = q_left + target_centroid.weight / new_total_weight;
        (self.scale_func)(q_right, self.compress_factor)
            - (self.scale_func)(q_left, self.compress_factor)
    }

    /// Estimate the value at a particular quantile
    /// # Arguments
    /// * `quantile` The quantile to estimate the value of
    pub fn interpolate(&self, quantile: f64) -> f64 {
        let total_count = self.total_weight();
        let mut current_quantile = 0.0;
        for i in 0..self.centroids.len() {
            // Quantile is located before this center of this centroid
            let new_quantile = current_quantile + (self.centroids[i].weight / (2.0 * total_count));
            if new_quantile > quantile {
                if i == 0 {
                    // Quantile is between the minimum value and the midpoint of the first centroid
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
                    // Quantile is bewteen the previous and current centroid
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
        // Quantile is between the midpoint of the last centroid and the maximum value
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

    /// Estimate the value at a particular quantile between two centroids
    /// # Arguments
    /// * `prev_centroid` Previous centroid
    /// * `current_centroid` Current centroid
    /// * `quantile` the quantile to estimate the value of
    /// * `current_quantile` the quantile on the left of the current centroid
    /// * `total_count` the total weight in the digest
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
        return quantile_proportion * (current_centroid.mean - prev_centroid.mean)
            + prev_centroid.mean;
    }

    /// Update the max and min values from a slice of new centroids
    /// # Arguments
    /// * `centroids` The centroids to extract max and min from
    fn update_limits(&mut self, centroids: &[Centroid]) {
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

/// Module containing scale functions for `TDigest`
pub mod scale_functions {
    use std::f64::consts::PI;
    pub fn k0(quantile: f64, comp_factor: f64) -> f64 {
        (quantile * comp_factor) / 2.0
    }

    pub fn inv_k0(scale: f64, comp_factor: f64) -> f64 {
        (scale * 2.0) / comp_factor
    }

    pub fn k1(quantile: f64, comp_factor: f64) -> f64 {
        (comp_factor / (2.0 * PI)) * (2.0 * quantile - 1.0).asin()
    }

    pub fn inv_k1(scale: f64, comp_factor: f64) -> f64 {
        (1.0 + (2.0 * PI * scale / comp_factor).sin()) / 2.0
    }

    pub fn k2(quantile: f64, comp_factor: f64) -> f64 {
        let n = 10.0;
        (comp_factor / (4.0 * (n / comp_factor).log10() + 24.0))
            * (quantile / (1.0 - quantile)).log10()
    }

    pub fn inv_k2(scale: f64, comp_factor: f64) -> f64 {
        let n: f64 = 10.0;
        let x =
            (10.0 as f64).powf((scale * (4.0 * (n / comp_factor).log10() + 24.0)) / comp_factor);
        return x / (1.0 + x);
    }

    pub fn k3(quantile: f64, comp_factor: f64) -> f64 {
        let n = 10.0;
        let factor = match quantile <= 0.5 {
            true => (2.0 * quantile).log10(),
            false => -(2.0 * (1.0 - quantile)).log10(),
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
