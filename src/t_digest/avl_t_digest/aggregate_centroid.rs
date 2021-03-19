use crate::t_digest::centroid::Centroid;
use num_traits::Float;
use std::ops::Add;

/// Centroid augmented with aggregate count for use in the AVL Tree T Digest
#[derive(Copy, Clone, Debug)]
pub struct AggregateCentroid<F>
where
    F: Float,
{
    /// Centroid being aggregated
    pub centroid: Centroid<F>,
    /// Sum of weights below this node
    pub aggregate_count: F,
}

impl<F> AggregateCentroid<F>
where
    F: Float,
{
    pub fn new(mean: F, weight: F) -> Self {
        Self {
            centroid: Centroid::new(mean, weight),
            aggregate_count: F::from(0.0).unwrap(),
        }
    }
}

impl<F> Add for AggregateCentroid<F>
where
    F: Float,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let new_centroid = self.centroid + other.centroid;
        Self::from(new_centroid)
    }
}

impl<F> From<Centroid<F>> for AggregateCentroid<F>
where
    F: Float,
{
    fn from(centroid: Centroid<F>) -> Self {
        Self {
            centroid,
            aggregate_count: F::from(0.0).unwrap(),
        }
    }
}

impl<F> Default for AggregateCentroid<F>
where
    F: Float,
{
    fn default() -> Self {
        Self::from(Centroid::default())
    }
}
