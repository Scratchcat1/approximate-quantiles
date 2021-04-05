use crate::t_digest::avl_t_digest::aggregate_centroid::AggregateCentroid;
use crate::t_digest::avl_t_digest::int_avl_tree_store::IntAVLTreeStore;
use num_traits::{Float, NumAssignOps};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct TreeCentroidStore<F>
where
    F: Float,
{
    centroids: Vec<AggregateCentroid<F>>,
}

impl<F> IntAVLTreeStore<AggregateCentroid<F>> for TreeCentroidStore<F>
where
    F: Float + NumAssignOps,
{
    fn with_capacity(capacity: u32) -> Self {
        Self {
            centroids: Vec::with_capacity(capacity as usize),
        }
    }

    fn resize(&mut self, new_capacity: u32) {
        self.centroids
            .resize(new_capacity as usize, AggregateCentroid::default());
    }

    fn merge(&mut self, _node: u32, _item: AggregateCentroid<F>) {
        unimplemented!();
    }

    fn copy(&mut self, node: u32, item: AggregateCentroid<F>) {
        self.centroids[node as usize] = item;
    }

    fn read(&self, node: u32) -> AggregateCentroid<F> {
        self.centroids[node as usize]
    }

    fn compare(&self, node: u32, item: AggregateCentroid<F>) -> Ordering {
        item.centroid
            .mean
            .partial_cmp(&self.centroids[node as usize].centroid.mean)
            .unwrap()
    }

    fn fix_aggregates(&mut self, node: u32, left: u32, right: u32) {
        let aggregate_count = self.centroids[node as usize].centroid.weight;
        let left_agg = self.centroids[left as usize].aggregate_count;

        let right_agg = self.centroids[right as usize].aggregate_count;

        self.centroids[node as usize].aggregate_count = aggregate_count + left_agg + right_agg;
    }
}
