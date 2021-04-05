use crate::t_digest::avl_t_digest::aggregate_centroid::AggregateCentroid;
use crate::t_digest::avl_t_digest::int_avl_tree::IntAVLTree;
use crate::t_digest::avl_t_digest::int_avl_tree_store::IntAVLTreeStore;
use crate::t_digest::avl_t_digest::tree_centroid_store::TreeCentroidStore;
use crate::t_digest::avl_t_digest::{node_id_to_option, NIL};
use crate::t_digest::centroid::Centroid;
use crate::traits::{Digest, OwnedSize};
use num_traits::{Float, NumAssignOps};
use rand;
use rand::{RngCore, SeedableRng};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct AVLTreeDigest<F, G, T>
where
    F: Fn(T, T, T) -> T,
    G: Fn(T, T, T) -> T,
    T: Float + NumAssignOps,
{
    /// Centroids tree
    pub tree: IntAVLTree<TreeCentroidStore<T>, AggregateCentroid<T>>,
    /// Compression factor to adjust the number of centroids to keep
    pub compress_factor: T,
    /// Scale function to map a quantile to a unit-less value to limit the size of a centroid
    pub scale_func: F,
    /// Function to invert the scale function
    pub inverse_scale_func: G,
    /// Keeps track of the minimum value observed
    pub min: T,
    /// Keeps track of the maximum value observed
    pub max: T,
    /// Total weight of centroids seen
    pub count: T,
}

impl<F, G, T> Digest<T> for AVLTreeDigest<F, G, T>
where
    F: Fn(T, T, T) -> T,
    G: Fn(T, T, T) -> T,
    T: Float + NumAssignOps,
{
    fn add(&mut self, mean: T) {
        self.add_centroid(Centroid::new(mean, T::from(1.0).unwrap()));
    }
    fn add_buffer(&mut self, items: &[T]) {
        for item in items {
            self.add_centroid(Centroid::new(*item, T::from(1.0).unwrap()));
        }
    }
    fn est_quantile_at_value(&mut self, val: T) -> T {
        if self.tree.size() == 0 {
            return T::from(f64::NAN).unwrap();
        } else if self.tree.size() == 1 {
            let result = match val
                .partial_cmp(
                    &self
                        .tree
                        .get_store()
                        .read(self.tree.first(self.tree.get_root()).unwrap())
                        .centroid
                        .mean,
                )
                .unwrap()
            {
                Ordering::Less => 0.0,
                Ordering::Greater => 1.0,
                Ordering::Equal => 0.5,
            };
            return T::from(result).unwrap();
        } else {
            if val < self.min {
                return T::from(0.0).unwrap();
            } else if val == self.min {
                // we have one or more centroids == val, treat them as one
                // dw will accumulate the weight of all of the centroids at val
                let mut dw = T::from(0.0).unwrap();
                for node in self.tree.into_iter() {
                    let agg_centroid = self.tree.get_store().read(node);
                    if agg_centroid.centroid.mean != val {
                        break;
                    }
                    dw += agg_centroid.centroid.weight;
                }

                return dw / (T::from(2.0).unwrap() * self.count);
            }

            if val > self.max {
                return T::from(1.0).unwrap();
            } else if val == self.max {
                // Same as for min, keep moving back to cover all centroids where mean == val == max
                let mut ix = self.tree.last(self.tree.get_root()).unwrap();
                let mut dw = T::from(0.0).unwrap();
                while ix != NIL && self.tree.get_store().read(ix).centroid.mean == val {
                    dw += self.tree.get_store().read(ix).centroid.weight;
                    ix = self.tree.prev(ix).unwrap();
                }
                return (self.count - dw / T::from(2.0).unwrap()) / self.count;
            }

            assert!(val < self.max);

            let first = self.tree.first(self.tree.get_root()).unwrap();
            let first_centroid = self.tree.get_store().read(first).centroid;
            // TODO this val > min should already be covered
            if val > self.min && val < first_centroid.mean {
                return self.interpolate_tail(val, first_centroid, self.min);
            }

            let last = self.tree.last(self.tree.get_root()).unwrap();
            let last_centroid = self.tree.get_store().read(last).centroid;
            // TODO this val < max should already be covered
            if val < self.max && val > last_centroid.mean {
                return T::from(1.0).unwrap() - self.interpolate_tail(val, last_centroid, self.max);
            }

            assert!(self.tree.size() >= 2);
            assert!(val >= first_centroid.mean);
            assert!(val <= last_centroid.mean);

            let mut it = self.tree.into_iter();
            let node_a = it.next().unwrap();
            let a = self.tree.get_store().read(node_a).centroid;
            let mut a_mean = a.mean;
            let mut a_weight = a.weight;

            if val == a_mean {
                return a.weight / (T::from(2.0).unwrap() * self.count);
            }

            // b is the lookahead to the next centroid
            let node_b = it.next().unwrap();
            let b = self.tree.get_store().read(node_b).centroid;
            let mut b_mean = b.mean;
            let mut b_weight = b.weight;

            assert!(b.mean >= a.mean);
            let mut weight_so_far = T::from(0.0).unwrap();

            while b_weight > T::from(0.0).unwrap() {
                assert!(val > a_mean);
                if val == b_mean {
                    weight_so_far += a_weight;
                    for node in it {
                        let b = self.tree.get_store().read(node).centroid;
                        if val == b_mean {
                            b_mean += b.weight;
                        } else {
                            break;
                        }
                    }
                    return (weight_so_far + b_weight / T::from(2.0).unwrap()) / self.count;
                }

                assert!(val < b_mean || val > b_mean);

                if val < b_mean {
                    // Strictly between a and b
                    assert!(a_mean < b_mean);

                    if a_weight == T::from(1).unwrap() {
                        if b_weight == T::from(1).unwrap() {
                            return (weight_so_far + T::from(1.0).unwrap()) / self.count;
                        } else {
                            let partial_weight = (val - a_mean) / (b_mean - a_mean) * b_weight
                                / T::from(2.0).unwrap();
                            return (weight_so_far + T::from(1.0).unwrap() + partial_weight)
                                / self.count;
                        }
                    } else if b_weight == T::from(1).unwrap() {
                        let partial_weight =
                            (val - a_mean) / (b_mean - a_mean) * a_weight / T::from(2.0).unwrap();
                        return (weight_so_far + a_weight / T::from(2.0).unwrap() + partial_weight)
                            / self.count;
                    } else {
                        let partial_weight =
                            (val - a_mean) / (b_mean - a_mean) * (a_weight + b_weight);
                        return (weight_so_far + a_weight / T::from(2.0).unwrap() + partial_weight)
                            / self.count;
                    }
                }

                weight_so_far += a_weight;
                assert!(val > b_mean);

                if let Some(node) = it.next() {
                    a_mean = b_mean;
                    a_weight = b_weight;

                    let b = self.tree.get_store().read(node).centroid;
                    b_mean = b.mean;
                    b_weight = b.weight;
                } else {
                    b_weight = T::from(0.0).unwrap();
                }
            }
            panic!("Ran out of centroids");
        }
    }
    fn est_value_at_quantile(&mut self, _: T) -> T {
        todo!()
    }
    fn count(&self) -> u64 {
        todo!()
    }
}

impl<F, G, T> AVLTreeDigest<F, G, T>
where
    F: Fn(T, T, T) -> T,
    G: Fn(T, T, T) -> T,
    T: Float + NumAssignOps,
{
    pub fn new(scale_func: F, inverse_scale_func: G, compress_factor: T) -> Self {
        Self {
            tree: IntAVLTree::new(16),
            compress_factor,
            scale_func,
            inverse_scale_func,
            min: T::max_value(),
            max: T::min_value(),
            count: T::from(0.0).unwrap(),
        }
    }

    fn add_centroid(&mut self, c: Centroid<T>) {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(31415); // TODO perhaps randomise this if the performance is alright
        let agg_c = AggregateCentroid::from(c);

        if c.mean < self.min {
            self.min = c.mean;
        }
        if c.mean > self.max {
            self.max = c.mean;
        }

        let mut start = self.tree.floor(agg_c);
        if start.is_none() {
            start = self.tree.first(self.tree.get_root());
        }
        if start.is_none() {
            // Ensure the tree is empty
            assert!(self.tree.size() == 0);
            self.tree.add_by(agg_c);
            self.count += c.weight;
        } else {
            let mut start = start.unwrap();

            let mut min_distance = T::max_value();
            let mut last_neighbour = NIL;
            let mut neighbour = start;

            while neighbour != NIL {
                let dist = (self.tree.get_store().read(neighbour).centroid.mean - c.mean).abs();
                if dist < min_distance {
                    start = neighbour;
                    min_distance = dist;
                } else if dist > min_distance {
                    // as soon as dist increases, we have passed the nearest neighbour and can quit
                    last_neighbour = neighbour;
                    break;
                }
                neighbour = self.tree.next(neighbour).unwrap_or(NIL);
            }

            let mut closest = NIL;
            let mut n = T::from(0.0).unwrap();
            let mut neighbour = start;
            while neighbour != last_neighbour {
                let n_centroid = self.tree.get_store().read(neighbour);
                assert!(min_distance == (n_centroid.centroid.mean - c.mean).abs());
                let q0 = self.head_sum(neighbour) / self.count;
                let q1 = q0 + n_centroid.centroid.weight / self.count;
                let k = self.count
                    * T::min(
                        (self.scale_func)(q0, self.compress_factor, self.count),
                        (self.scale_func)(q1, self.compress_factor, self.count),
                    );

                if n_centroid.centroid.weight + c.weight <= k {
                    n += T::from(1.0).unwrap();
                    if (T::from(rng.next_u32()).unwrap() / T::from(u32::max_value()).unwrap())
                        < T::from(1.0).unwrap() / n
                    {
                        closest = neighbour;
                    }
                }
                neighbour = self.tree.next(neighbour).unwrap_or(NIL);
            }

            if closest == NIL {
                self.tree.add_by(agg_c);
            } else {
                let closest_centroid = self.tree.get_store().read(closest).centroid;
                self.update(
                    closest,
                    AggregateCentroid::from(closest_centroid + c),
                    false,
                );
            }

            self.count += c.weight;

            if self.tree.size()
                > (T::from(20).unwrap() * self.compress_factor)
                    .to_u32()
                    .unwrap()
            {
                self.compress();
            }
        }
    }

    fn compress(&mut self) {
        if self.tree.size() <= 1 {
            return;
        }

        let mut n0 = T::from(0.0).unwrap();
        let mut k0 =
            self.count * (self.scale_func)(n0 / self.count, self.compress_factor, self.count);
        let mut node = self.tree.first(self.tree.get_root()).expect(
            "A tree with size greater than 1 should have a first node
            ",
        );
        let mut w0 = self.tree.get_store().read(node).centroid.weight;

        let mut n1 = n0 + w0;
        let mut w1 = T::from(0.0).unwrap();
        while node != NIL {
            let mut after = self.tree.next(node);
            while let Some(after_node) = after {
                let node_centroid = self.tree.get_store().read(node);
                let after_centroid = self.tree.get_store().read(after_node);
                w1 = after_centroid.centroid.weight;
                let k1 = self.count
                    * (self.scale_func)((n1 + w1) / self.count, self.compress_factor, self.count);
                if w0 + w1 > T::min(k0, k1) {
                    break;
                } else {
                    let new_centroid = Centroid::new(node_centroid.centroid.mean, w0)
                        + Centroid::new(after_centroid.centroid.mean, w1);
                    self.update(node, AggregateCentroid::from(new_centroid), true);

                    let tmp = self.tree.next(after_node).unwrap_or(NIL);
                    self.tree.remove(after_node);
                    after = node_id_to_option(tmp);
                    n1 += w1;
                    w0 += w1;
                }
            }

            node = after.unwrap_or(NIL);
            if node != NIL {
                n0 = n1;
                k0 = (self.scale_func)(n0 / self.count, self.compress_factor, self.count);
                w0 = w1;
                n1 = n0 + w0;
            }
        }
    }

    fn update(&mut self, node: u32, new_centroid: AggregateCentroid<T>, force_in_place: bool) {
        let old_agg_centroid = self.tree.get_store().read(node);
        // TODO is this needed? to update aggregate counts??
        // new_centroid.aggregate_count = old_agg_centroid.aggregate_count;
        // Update in place if possible, if the centroid would be in the same position
        if old_agg_centroid.centroid.mean == new_centroid.centroid.mean || force_in_place {
            self.tree.get_mut_store().copy(node, new_centroid);
        } else {
            self.tree.update_by(node, new_centroid);
        }
    }

    /// Returns the last node id such that the sum of weights below that centroid is less than or equal to `sum`.
    fn floor_sum(&self, mut sum: T) -> Option<u32> {
        let mut floor = NIL;
        let mut node = self.tree.get_root();
        while node != NIL {
            let left = self.tree.get_left(node);
            let left_centroid = self.tree.get_store().read(left);
            let left_weight = left_centroid.aggregate_count;
            match left_weight.partial_cmp(&sum).unwrap() {
                Ordering::Less | Ordering::Equal => {
                    floor = node;
                    sum -= left_weight + left_centroid.centroid.weight;
                    node = self.tree.get_right(node);
                }
                Ordering::Greater => {
                    node = self.tree.get_left(node);
                }
            }
        }

        node_id_to_option(floor)
    }

    /// Returns the sum of weights for nodes to the left of `node`
    fn head_sum(&self, node: u32) -> T {
        let left = self.tree.get_left(node);
        let mut sum = self.tree.get_store().read(left).aggregate_count;

        let mut n = node;
        let mut p = self.tree.get_parent(node);

        while p != NIL {
            if n == self.tree.get_right(p) {
                let left_parent = self.tree.get_left(p);
                sum += self.tree.get_store().read(p).centroid.weight
                    + self.tree.get_store().read(left_parent).aggregate_count;
            }
            n = p;
            p = self.tree.get_parent(n);
        }

        sum
    }

    fn interpolate_tail(&self, val: T, centroid: Centroid<T>, extreme_value: T) -> T {
        assert!(centroid.weight > T::from(1).unwrap());

        if centroid.weight == T::from(2.0).unwrap() {
            return T::from(1.0).unwrap() / self.count;
        } else {
            let weight = centroid.weight / T::from(2.0).unwrap() - T::from(1).unwrap();

            let partial_weight = (extreme_value - val) / (extreme_value - centroid.mean) * weight;

            return (partial_weight + T::from(1).unwrap()) / self.count;
        }
    }

    pub fn check_aggregates(&self) {
        self.check_node_aggregates(self.tree.get_root());
    }

    pub fn check_node_aggregates(&self, node: u32) {
        let store = self.tree.get_store();
        let node_centroid = store.read(node);
        let left_centroid = store.read(self.tree.get_left(node));
        let right_centroid = store.read(self.tree.get_right(node));
        assert!(
            node_centroid.aggregate_count
                == node_centroid.centroid.weight
                    + left_centroid.aggregate_count
                    + right_centroid.aggregate_count
        );

        if node != NIL {
            self.check_node_aggregates(self.tree.get_left(node));
            self.check_node_aggregates(self.tree.get_right(node));
        }
    }
}

#[cfg(test)]
mod test {
    use crate::t_digest::avl_t_digest::avl_tree_digest::AVLTreeDigest;
    use crate::t_digest::centroid::Centroid;
    use crate::t_digest::scale_functions::{inv_k0, inv_k2, k0, k1, k1_max, k2};
    use crate::traits::Digest;
    use crate::util::linear_digest::LinearDigest;
    use crate::util::{gen_asc_vec, gen_uniform_vec};
    use approx::assert_relative_eq;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn avl_uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = AVLTreeDigest::new(&k1_max, &k1_max, 2000.0);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);
        digest.check_aggregates();

        assert_relative_eq!(
            digest.est_quantile_at_value(0.0),
            linear_digest.est_quantile_at_value(0.0)
        );
        println!(
            "{} {}",
            digest.est_quantile_at_value(1.0),
            linear_digest.est_quantile_at_value(1.0)
        );
        let x = digest.est_quantile_at_value(1.0);
        assert_relative_eq!(
            x / linear_digest.est_quantile_at_value(1.0),
            1.0,
            epsilon = 0.04
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(10.0) / linear_digest.est_quantile_at_value(10.0),
            1.0,
            epsilon = 0.02
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(250.0) / linear_digest.est_quantile_at_value(250.0),
            1.0,
            epsilon = 0.01
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
    }
}
