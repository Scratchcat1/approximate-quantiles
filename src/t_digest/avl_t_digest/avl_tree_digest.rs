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
    fn add_buffer(&mut self, _: &[T]) {
        todo!()
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
        }

        todo!()
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
            assert_eq!(self.tree.size(), 0);
            self.tree.add_by(agg_c);
        } else {
            let start = start.unwrap();

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
            }

            let mut closest = NIL;
            let mut n = T::from(0.0).unwrap();
            let mut neighbour = start;
            while neighbour != last_neighbour {
                let n_centroid = self.tree.get_store().read(neighbour);
                assert_eq!(min_distance, (n_centroid.centroid.mean - c.mean));
                let q0 = self.head_sum(neighbour) / self.count;
                let q1 = q0 + n_centroid.centroid.weight / self.count;
                let k = self.count
                    * T::min(
                        (self.scale_func)(self.compress_factor, q0, self.count),
                        (self.scale_func)(self.compress_factor, q1, self.count),
                    );

                if n_centroid.centroid.weight + c.weight <= k {
                    n += T::from(1.0).unwrap();
                    if (T::from(rng.next_u32()).unwrap() / T::from(u32::max_value()).unwrap())
                        < T::from(1.0).unwrap() / n
                    {
                        closest = neighbour;
                    }
                }
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
                self.count += c.weight;
            }

            // The reference implementation does have this second addition but it seems incorrect, TODO
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
            self.count * (self.scale_func)(self.compress_factor, n0 / self.count, self.count);
        let mut node = self.tree.first(self.tree.get_root()).expect(
            "A tree with size greater than 1 should have a first node
            ",
        );
        let mut w0 = self.tree.get_store().read(node).centroid.weight;

        let mut n1 = n0 + w0;
        let mut w1 = T::from(0.0).unwrap();
        while node != NIL {
            let after = self.tree.next(node);
            while let Some(after) = after {
                let node_centroid = self.tree.get_store().read(node);
                let after_centroid = self.tree.get_store().read(after);
                w1 = after_centroid.centroid.weight;
                let k1 = self.count
                    * (self.scale_func)(self.compress_factor, (n1 + w1) / self.count, self.count);
                if w0 + w1 > T::min(k0, k1) {
                    break;
                } else {
                    let new_centroid = node_centroid + after_centroid;
                    self.update(node, new_centroid, true);

                    let tmp = self.tree.next(after).unwrap_or(NIL);
                    self.tree.remove(after);
                    after = tmp;
                    n1 += w1;
                    w0 += w1;
                }
            }

            node = after.unwrap_or(NIL);
            if node != NIL {
                n0 = n1;
                k0 = (self.scale_func)(self.compress_factor, n0 / self.count, self.count);
                w0 = w1;
                n1 = n1 + w0;
            }
        }
    }

    fn update(&mut self, node: u32, new_centroid: AggregateCentroid<T>, force_in_place: bool) {
        let old_agg_centroid = self.tree.get_store().read(node);
        // TODO is this needed? to update aggregate counts??
        new_centroid.aggregate_count = old_agg_centroid.aggregate_count;
        // Update in place if possible, if the centroid would be in the same position
        if old_agg_centroid.centroid.mean == new_centroid.centroid.mean || force_in_place {
            self.tree.get_mut_store().copy(node, new_centroid);
        } else {
            self.tree.add_by(new_centroid);
        }
    }

    /// Returns the last node id such that the sum of weights below that centroid is less than or equal to `sum`.
    fn floor_sum(&self, mut sum: T) -> Option<u32> {
        let floor = NIL;
        let mut node = self.tree.get_root();
        while node != NIL {
            let left = self.tree.get_left(node);
            let left_centroid = self.tree.get_store().read(node);
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
                    + self.tree.get_store().read(p).aggregate_count;
            }
        }

        sum
    }
}
