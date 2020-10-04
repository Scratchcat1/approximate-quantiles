#![allow(dead_code)]
use crate::t_digest::centroid::Centroid;
use crate::traits::Digest;
use crate::util::weighted_average;

pub struct TDigest<F, G>
where
    F: Fn(f64, f64, f64) -> f64,
    G: Fn(f64, f64, f64) -> f64,
{
    /// Vector of centroids
    pub centroids: Vec<Centroid>,
    /// Compression factor to adjust the number of centroids to keep
    pub compress_factor: f64,
    /// Scale function to map a quantile to a unit-less value to limit the size of a centroid
    pub scale_func: F,
    /// Function to invert the scale function
    pub inverse_scale_func: G,
    /// Keeps track of the minimum value observed
    pub min: f64,
    /// Keeps track of the maximum value observed
    pub max: f64,
}

impl<F, G> Digest for TDigest<F, G>
where
    F: Fn(f64, f64, f64) -> f64,
    G: Fn(f64, f64, f64) -> f64,
{
    fn add(&mut self, item: f64) {
        self.add_centroid_buffer(vec![Centroid {
            mean: item,
            weight: 1.0,
        }]);
    }

    fn add_buffer(&mut self, buffer: Vec<f64>) {
        self.add_centroid_buffer(
            buffer
                .into_iter()
                .map(|item| Centroid {
                    mean: item,
                    weight: 1.0,
                })
                .collect::<Vec<Centroid>>(),
        );
    }

    fn est_quantile_at_value(&mut self, item: f64) -> f64 {
        if item <= self.min {
            return 0.0;
        } else if item >= self.max {
            return 1.0;
        }
        let total_count = self.total_weight();
        let mut current_quantile = 0.0;
        for i in 0..self.centroids.len() {
            if item < self.centroids[i].mean {
                if i == 0 {
                    let prev_centroid = Centroid {
                        mean: self.min,
                        weight: 1.0,
                    };
                    return self.interpolate_centroids_quantile(
                        &prev_centroid,
                        &self.centroids[i],
                        item,
                        prev_centroid.weight / (2.0 * total_count),
                        total_count,
                    );
                } else {
                    // item is between the previous and current centroid
                    let prev_centroid = &self.centroids[i - 1];
                    return self.interpolate_centroids_quantile(
                        prev_centroid,
                        &self.centroids[i],
                        item,
                        current_quantile,
                        total_count,
                    );
                }
            }
            current_quantile += self.centroids[i].weight / total_count;
        }
        // item is between the midpoint of the last centroid and the maximum value
        let curr_centroid = Centroid {
            mean: self.max,
            weight: 1.0,
        };
        return self.interpolate_centroids_quantile(
            &self.centroids[self.centroids.len() - 1],
            &curr_centroid,
            item,
            current_quantile,
            total_count,
        );
    }

    fn est_value_at_quantile(&mut self, target_quantile: f64) -> f64 {
        let total_weight = self.total_weight();
        let target_index = total_weight * target_quantile;

        if target_index < 1.0 {
            return self.min;
        }

        let first = &self.centroids[0];
        if first.weight > 1.0 && target_index < first.weight / 2.0 {
            return self.min
                + (target_index - 1.0) / (first.weight / 2.0 - 1.0) * (first.mean - self.min);
        }

        if target_index > total_weight - 1.0 {
            return self.max;
        }

        let last = self.centroids.last().unwrap();
        if last.weight > 1.0 && total_weight - target_index <= last.weight / 2.0 {
            return self.max - (total_weight - target_index - 1.0) / (last.weight / 2.0 - 1.0);
        }

        let mut curr_weight = first.weight / 2.0;
        for i in 0..(total_weight as usize) {
            let curr = &self.centroids[i];
            let next = &self.centroids[i + 1];
            let dw = (curr.weight + next.weight) / 2.0;

            if curr_weight + dw > target_index {
                if curr.weight == 1.0 && target_index - curr_weight < 0.5 {
                    return curr.mean;
                }

                if next.weight == 1.0 && curr_weight + dw - target_index <= 0.5 {
                    return next.mean;
                }

                let z1 = target_index - curr_weight - 0.5;
                let z2 = curr_weight + dw - target_index - 0.5;
                return weighted_average(curr.mean, z2, next.mean, z1);
            }

            curr_weight += dw;
        }

        let z1 = target_index - total_weight - last.weight / 2.0;
        let z2 = last.weight / 2.0 - z1;
        return weighted_average(last.mean, z1, self.max, z2);
    }
}

impl<F, G> TDigest<F, G>
where
    F: Fn(f64, f64, f64) -> f64,
    G: Fn(f64, f64, f64) -> f64,
{
    /// Returns a new `TDigest`
    /// # Arguments
    ///
    /// * `scale_func` Scale function
    /// * `inverse_scale_func` Inverse scale function
    /// * `compress_factor` Compression factor
    pub fn new(scale_func: F, inverse_scale_func: G, compress_factor: f64) -> TDigest<F, G> {
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
    pub fn add_centroid_buffer(&mut self, mut buffer: Vec<Centroid>) {
        // Merge the digest centroids into the buffer since normally |buffer| > |self.centroids|
        buffer.extend(self.centroids.clone());
        // Nothing to merge, exit
        if buffer.len() == 0 {
            return;
        }
        buffer.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        let num_elements: f64 = buffer.iter().map(|c| c.weight).sum();
        self.min = f64::min(self.min, buffer.first().unwrap().mean);
        self.max = f64::max(self.max, buffer.last().unwrap().mean);

        // Use weights instead of quantiles to minimise division in the main loop
        let mut w0 = 0.0;
        let get_w_limit = |w0| {
            (self.inverse_scale_func)(
                (self.scale_func)(w0 / num_elements, self.compress_factor, num_elements) + 1.0,
                self.compress_factor,
                num_elements,
            ) * num_elements
        };
        let mut w_size_limit = get_w_limit(w0);
        let mut new_centroids = Vec::new();

        let mut buffer_iter = buffer.into_iter();
        let first_centroid = buffer_iter.next().unwrap();

        let mut mergeable_weight = first_centroid.weight;
        let mut mergeable_sum = first_centroid.mean * first_centroid.weight;
        for next_centroid in buffer_iter {
            let new_w = mergeable_weight + next_centroid.weight;

            // If combined weight is below the limit merge the centroids
            if new_w <= w_size_limit {
                mergeable_weight = new_w;
                mergeable_sum += next_centroid.mean * next_centroid.weight;
            } else {
                // Combined weight exceeds limit, add the current centroid to the vector and calculate the new limit
                w0 += mergeable_weight;
                new_centroids.push(Centroid {
                    mean: mergeable_sum / mergeable_weight,
                    weight: mergeable_weight,
                });
                w_size_limit = get_w_limit(w0) - w0;
                mergeable_weight = next_centroid.weight;
                mergeable_sum = next_centroid.mean * next_centroid.weight;
            }
        }
        new_centroids.push(Centroid {
            mean: mergeable_sum / mergeable_weight,
            weight: mergeable_weight,
        });
        self.centroids = new_centroids;
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
        (self.scale_func)(q_right, self.compress_factor, new_total_weight)
            - (self.scale_func)(q_left, self.compress_factor, new_total_weight)
    }

    /// Estimate the quantile of a particular value between two centroids
    /// # Arguments
    /// * `prev_centroid` Previous centroid
    /// * `current_centroid` Current centroid
    /// * `target_value` the value to estimate the quantile of
    /// * `current_quantile` the quantile on the left of the current centroid
    /// * `total_count` the total weight in the digest
    fn interpolate_centroids_quantile(
        &self,
        prev_centroid: &Centroid,
        current_centroid: &Centroid,
        target_value: f64,
        current_quantile: f64,
        total_count: f64,
    ) -> f64 {
        let prev_quantile = current_quantile - (prev_centroid.weight / (2.0 * total_count));
        let next_quantile = current_quantile + (current_centroid.weight / (2.0 * total_count));
        let proportion =
            (target_value - prev_centroid.mean) / (current_centroid.mean - prev_centroid.mean);
        return proportion * (next_quantile - prev_quantile) + prev_quantile;
    }

    /// Update the max and min values from a slice of new centroids
    /// # Arguments
    /// * `centroids` The centroids to extract max and min from
    fn update_limits(&mut self, centroids: &[Centroid]) {
        self.min = f64::min(
            centroids
                .iter()
                .map(|x| x.mean)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or_else(|| f64::INFINITY),
            self.min,
        );

        self.max = f64::max(
            centroids
                .iter()
                .map(|x| x.mean)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or_else(|| f64::NEG_INFINITY),
            self.max,
        );
    }
}

#[cfg(test)]
mod test {
    use crate::t_digest::centroid::Centroid;
    use crate::t_digest::scale_functions::{inv_k0, inv_k1, inv_k2, k0, k1, k2};
    use crate::t_digest::t_digest::TDigest;
    use crate::traits::Digest;
    use crate::util::linear_digest::LinearDigest;
    use approx::assert_relative_eq;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn add_buffer_with_single_centroid() {
        let buffer = vec![Centroid {
            mean: 1.0,
            weight: 1.0,
        }];
        let mut digest = TDigest::new(&k0, &inv_k0, 1.0);
        digest.add_centroid_buffer(buffer);

        assert_eq!(digest.centroids.len(), 1);
        assert_eq!(digest.centroids[0].mean, 1.0);
        assert_eq!(digest.centroids[0].weight, 1.0);
        assert_eq!(digest.total_weight(), 1.0);
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
        digest.add_centroid_buffer(buffer);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0, epsilon = 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0);
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
        digest.add_centroid_buffer(buffer);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0, epsilon = 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }

    #[test]
    fn add_buffer_uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = TDigest::new(&k2, &inv_k2, 2000.0);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(buffer.clone());
        linear_digest.add_buffer(buffer.clone());

        println!("{}", digest.centroids.len());
        assert_relative_eq!(
            digest.est_value_at_quantile(0.0) / linear_digest.est_value_at_quantile(0.0),
            1.0,
            epsilon = 0.0005
        );
        assert_relative_eq!(
            digest.est_value_at_quantile(0.001) / linear_digest.est_value_at_quantile(0.001),
            1.0,
            epsilon = 0.0075
        );
        assert_relative_eq!(
            digest.est_value_at_quantile(0.01) / linear_digest.est_value_at_quantile(0.01),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_value_at_quantile(0.25) / linear_digest.est_value_at_quantile(0.25),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_value_at_quantile(0.5) / linear_digest.est_value_at_quantile(0.5),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_value_at_quantile(0.75) / linear_digest.est_value_at_quantile(0.75),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_value_at_quantile(1.0) / linear_digest.est_value_at_quantile(1.0),
            1.0,
            epsilon = 0.005
        );
        assert_eq!(digest.total_weight(), linear_digest.values.len() as f64);
    }

    #[test]
    fn add_buffer_uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 2000.0);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(buffer.clone());
        linear_digest.add_buffer(buffer.clone());

        println!("{}", digest.centroids.len());
        assert_relative_eq!(
            digest.est_quantile_at_value(0.0),
            linear_digest.est_quantile_at_value(0.0)
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(1.0) / linear_digest.est_quantile_at_value(1.0),
            1.0,
            epsilon = 0.0075
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(10.0) / linear_digest.est_quantile_at_value(10.0),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(250.0) / linear_digest.est_quantile_at_value(250.0),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(500.0) / linear_digest.est_quantile_at_value(500.0),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(750.0) / linear_digest.est_quantile_at_value(750.0),
            1.0,
            epsilon = 0.005
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(1000.0) / linear_digest.est_quantile_at_value(1000.0),
            1.0,
            epsilon = 0.005
        );
        assert_eq!(digest.total_weight(), linear_digest.values.len() as f64);
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
        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0, epsilon = 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0);
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
        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0, epsilon = 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }

    #[test]
    fn est_quantile_at_value() {
        let buffer = (0..1001)
            .map(|x| Centroid {
                mean: -500.0 + x as f64,
                weight: 1.0,
            })
            .collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 100.0);
        digest.add_centroid_buffer(buffer);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(digest.est_quantile_at_value(-500.0), 0.0);
        assert_relative_eq!(digest.est_quantile_at_value(-250.0), 0.25, epsilon = 0.001);
        assert_relative_eq!(digest.est_quantile_at_value(0.0), 0.5, epsilon = 0.001);
        assert_relative_eq!(digest.est_quantile_at_value(250.0), 0.75, epsilon = 0.001);
        assert_relative_eq!(digest.est_quantile_at_value(500.0), 1.0);
    }

    #[test]
    fn est_value_at_quantile_singleton_centroids() {
        let mut digest = TDigest::new(&k0, &inv_k0, 50.0);
        digest.add_buffer(vec![1.0, 2.0, 8.0, 0.5]);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.5);
        assert_relative_eq!(digest.est_value_at_quantile(0.24), 0.5);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.49), 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.50), 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.74), 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 8.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 8.0);
        assert_eq!(digest.centroids.len(), 4);
        assert_eq!(digest.total_weight(), 4.0);
    }
}
