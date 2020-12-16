#![allow(dead_code)]
use crate::t_digest::centroid::Centroid;
use crate::traits::Digest;
use crate::util::keyed_sum_tree::KeyedSumTree;
use crate::util::weighted_average;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

#[derive(Clone)]
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

    fn add_buffer(&mut self, buffer: &[f64]) {
        self.add_centroid_buffer(
            buffer
                .iter()
                .map(|item| Centroid {
                    mean: *item,
                    weight: 1.0,
                })
                .collect::<Vec<Centroid>>(),
        );
    }

    fn est_quantile_at_value(&mut self, item: f64) -> f64 {
        // From https://github.com/tdunning/t-digest/blob/cba43e734ffe226efc7829b622459a6efb64e1e1/core/src/main/java/com/tdunning/math/stats/MergingDigest.java#L549
        if self.centroids.len() == 0 {
            return f64::NAN;
        } else if self.centroids.len() == 1 {
            let width = self.max - self.min;
            return if item < self.min {
                0.0
            } else if item > self.max {
                1.0
            } else if item - self.min <= width {
                0.5
            } else {
                (item - self.min) / width
            };
        } else {
            if item < self.min {
                return 0.0;
            }
            if item > self.max {
                return 1.0;
            }

            let total_weight = self.total_weight();

            let first = &self.centroids[0];
            if item < first.mean {
                if first.mean - self.min > 0.0 {
                    if item == self.min {
                        return 0.5 / total_weight;
                    } else {
                        return (1.0
                            + (item - self.min) / (first.mean - self.min)
                                * (first.weight / 2.0 - 1.0))
                            / total_weight;
                    }
                } else {
                    return 0.0;
                }
            }

            let last = &self.centroids.last().unwrap();
            if item > last.mean {
                if self.max - last.mean > 0.0 {
                    if item == self.max {
                        return 1.0 - 0.5 / total_weight;
                    } else {
                        return 1.0
                            - ((1.0
                                + (self.max - item) / (self.max - last.mean)
                                    * (last.weight / 2.0 - 1.0))
                                / total_weight);
                    }
                } else {
                    return 1.0;
                }
            }

            let mut weight_so_far = 0.0;
            for i in 0..self.centroids.len() - 1 {
                if self.centroids[i].mean == item {
                    let mut dw = 0.0;

                    for j in i..self.centroids.len() {
                        if self.centroids[j].mean != item {
                            break;
                        }
                        dw += self.centroids[j].weight;
                    }
                    if self.centroids[i].weight == dw && dw == 1.0 {
                        // The value matches a single singleton centroid. Therefore there is no weight to the left to split in half.
                        return weight_so_far / total_weight;
                    }
                    return (weight_so_far + dw / 2.0) / total_weight;
                } else if self.centroids[i].mean <= item && item < self.centroids[i + 1].mean {
                    if self.centroids[i + 1].mean - self.centroids[i].mean > 0.0 {
                        let mut left_excluded_weight = 0.0;
                        let mut right_excluded_weight = 0.0;
                        if self.centroids[i].weight == 1.0 {
                            if self.centroids[i + 1].weight == 1.0 {
                                return (weight_so_far + 1.0) / total_weight;
                            } else {
                                left_excluded_weight = 0.5;
                            }
                        } else if self.centroids[i + 1].weight == 1.0 {
                            right_excluded_weight = 0.5;
                        }

                        let dw = (self.centroids[i].weight + self.centroids[i + 1].weight) / 2.0;

                        // Don't currently assert as weight is not enforced to be greater than 1
                        // assert!(dw > 1.0);
                        // assert!(left_excluded_weight + right_excluded_weight <= 0.5);
                        let left = self.centroids[i].mean;
                        let right = self.centroids[i + 1].mean;
                        let dw_no_singleton = dw - left_excluded_weight - right_excluded_weight;

                        // assert!(dw_no_singleton > dw / 2.0);
                        // assert!(right - left > 0.0);
                        let base =
                            weight_so_far + self.centroids[i].weight / 2.0 + left_excluded_weight;
                        // println!(
                        //     "le {} , re {}, dw {} base {} ratio {}",
                        //     left_excluded_weight,
                        //     right_excluded_weight,
                        //     dw,
                        //     base,
                        //     (item - left) / (right - left)
                        // );
                        return (base + dw_no_singleton * (item - left) / (right - left))
                            / total_weight;
                    } else {
                        let dw = (self.centroids[i].weight + self.centroids[i + 1].weight) / 2.0;
                        return (weight_so_far + dw) / total_weight;
                    }
                } else {
                    weight_so_far += self.centroids[i].weight;
                }
            }

            if item == last.mean {
                if last.weight == 1.0 {
                    // Process as singleton
                    return weight_so_far / total_weight;
                }
                return 1.0 - 0.5 / total_weight;
            } else {
                panic!("Illegal state, Fell through loop");
            }
        }
    }

    fn est_value_at_quantile(&mut self, target_quantile: f64) -> f64 {
        // From https://github.com/tdunning/t-digest/blob/cba43e734ffe226efc7829b622459a6efb64e1e1/core/src/main/java/com/tdunning/math/stats/MergingDigest.java#L687
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
        buffer.par_sort_unstable_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
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
        let mut total_weight = self.total_weight();
        for x in clusters {
            let close_centroids = self.find_closest_centroids(&x);
            match close_centroids {
                Some(indexes) => {
                    // Find the index of a centroid with space to merge the current centroid
                    // selecting the one with the minimum weight
                    let mut min_acceptable_index = None;
                    for index in indexes {
                        let new_centroid = &x + &self.centroids[index];
                        if self.k_size(&new_centroid, total_weight).abs() < 1.0 {
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
                            total_weight += x.weight;
                            merge.mean = (merge.mean * merge.weight + x.mean * x.weight)
                                / (merge.weight + x.weight);
                            merge.weight += x.weight;
                        }
                        None => {
                            // No suitable centroid in the digest was found, insert the current centroid into the digest
                            total_weight += x.weight;
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
                    total_weight += x.weight;
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
                self.add_buffer(&Vec::new());
            }
        }
        // Don't perform a final merge, this significantly improves performance while digest size is still bounded by the growth limit.
    }

    /// Add centroids to the digest via clustering
    ///
    /// # Arguments
    /// * `clusters` Centroids to add to the digest
    /// * `growth_limit` Factor to limit excessive growth of the digest by merging periodically
    pub fn add_cluster_tree(&mut self, clusters: Vec<Centroid>, growth_limit: f64) {
        self.update_limits(&clusters);
        let mut total_weight = self.total_weight();
        let mut rng = thread_rng();
        let mut cloned_centroids = self.centroids.clone();
        cloned_centroids.shuffle(&mut rng);
        let mut k_size_tree = KeyedSumTree::from(&cloned_centroids[..]);
        for x in clusters {
            let closest_centroids = k_size_tree.closest_keys(x.mean);
            match closest_centroids.is_empty() {
                false => {
                    // Find the centroid with space to merge the current centroid
                    // selecting the one with the minimum weight
                    let mut closest_acceptable_centroid = None;
                    for close_centroid in closest_centroids {
                        let new_mean = (&x + &close_centroid).mean;
                        if self
                            .k_size_from_weights(
                                new_mean,
                                k_size_tree.less_than_sum(new_mean).unwrap_or(0.0),
                                total_weight + x.weight,
                            )
                            .abs()
                            < 1.0
                        {
                            match &closest_acceptable_centroid {
                                None => closest_acceptable_centroid = Some(close_centroid),
                                Some(other) => {
                                    if other.mean * other.weight
                                        > close_centroid.mean * close_centroid.weight
                                    {
                                        closest_acceptable_centroid = Some(close_centroid)
                                    }
                                }
                            }
                        }
                    }

                    match closest_acceptable_centroid {
                        Some(closest_centroid) => {
                            // Merge the current centroid with the centroid in the digest
                            total_weight += x.weight;
                            k_size_tree.delete(closest_centroid.mean);

                            let mean = (closest_centroid.mean * closest_centroid.weight
                                + x.mean * x.weight)
                                / (closest_centroid.weight + x.weight);
                            let weight = closest_centroid.weight + x.weight;
                            k_size_tree.insert(mean, weight);
                        }
                        None => {
                            // No suitable centroid in the digest was found, insert the current centroid into the digest
                            k_size_tree.insert(x.mean, x.weight);
                            total_weight += x.weight;
                        }
                    }
                }
                true => {
                    // No suitable centroid in the digest was found, insert the current centroid into the digest
                    k_size_tree.insert(x.mean, x.weight);
                    total_weight += x.weight;
                }
            }

            // Prevent excess growth with particular insertion patterns by periodically merging
            if k_size_tree.size() > (growth_limit * self.compress_factor) as usize {
                self.centroids = k_size_tree.sorted_vec_key();
                self.add_buffer(&Vec::new());
                let mut cloned_centroids = self.centroids.clone();
                cloned_centroids.shuffle(&mut rng);
                k_size_tree = KeyedSumTree::from(&cloned_centroids[..]);
            }
        }
        self.centroids = k_size_tree.sorted_vec_key();
        // Don't perform a final merge, this significantly improves performance while digest size is still bounded by the growth limit.
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
    pub fn k_size(&self, target_centroid: &Centroid, total_weight: f64) -> f64 {
        let new_total_weight = total_weight + target_centroid.weight;

        // Calculate the left and right quartiles
        self.k_size_from_weights(
            target_centroid.weight,
            self.weight_left(target_centroid),
            new_total_weight,
        )
    }

    pub fn k_size_from_weights(&self, weight: f64, weight_left: f64, new_total_weight: f64) -> f64 {
        let q_left = weight_left / new_total_weight;
        let q_right = q_left + weight / new_total_weight;
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
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

        println!("{}", digest.centroids.len());
        assert_relative_eq!(
            digest.est_value_at_quantile(0.0) / linear_digest.est_value_at_quantile(0.0),
            1.0,
            epsilon = 0.00005
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
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

        println!("{:?}", digest.centroids);
        assert_relative_eq!(
            digest.est_quantile_at_value(0.0),
            linear_digest.est_quantile_at_value(0.0)
        );
        println!("{}", digest.est_quantile_at_value(1.0));
        assert_relative_eq!(
            digest.est_quantile_at_value(1.0) / linear_digest.est_quantile_at_value(1.0),
            1.0,
            epsilon = 0.0075
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(10.0) / linear_digest.est_quantile_at_value(10.0),
            1.0,
            epsilon = 0.001
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(250.0) / linear_digest.est_quantile_at_value(250.0),
            1.0,
            epsilon = 0.0005
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(500.0) / linear_digest.est_quantile_at_value(500.0),
            1.0,
            epsilon = 0.0005
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(750.0) / linear_digest.est_quantile_at_value(750.0),
            1.0,
            epsilon = 0.0005
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
        let buffer: Vec<f64> = (0..1000).map(|x| -500.0 + x as f64).collect();
        let mut digest = TDigest::new(&k1, &inv_k1, 100.0);
        digest.add_buffer(&buffer);

        let mut linear_digest = LinearDigest::new();
        linear_digest.add_buffer(&buffer);

        println!("{:?}", digest.centroids);
        println!("{:?}", &linear_digest.values[245..255]);
        println!(
            "{}, {}",
            digest.est_quantile_at_value(-250.0),
            linear_digest.est_quantile_at_value(-250.0)
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(-500.0),
            linear_digest.est_quantile_at_value(-500.0),
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(-250.0) / linear_digest.est_quantile_at_value(-250.0),
            1.0,
            epsilon = 0.00025
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(0.0) / linear_digest.est_quantile_at_value(0.0),
            1.0,
            epsilon = 0.00001
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(250.0) / linear_digest.est_quantile_at_value(250.0),
            1.0,
            epsilon = 0.00001
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(500.0) / linear_digest.est_quantile_at_value(500.0),
            1.0,
        );
    }

    #[test]
    fn est_value_at_quantile_singleton_centroids() {
        let mut digest = TDigest::new(&k0, &inv_k0, 50.0);
        digest.add_buffer(&vec![1.0, 2.0, 8.0, 0.5]);

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
