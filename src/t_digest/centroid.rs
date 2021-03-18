use num_traits::Float;
use std::ops::Add;

/// Weighted value used in `TDigest`
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Centroid<F>
where
    F: Float,
{
    /// Mean of the centroid
    pub mean: F,
    /// Weight of the centroid
    pub weight: F,
}

// unsafe impl<F> Send for Centroid<F> where F: Float {}

impl<F> Centroid<F>
where
    F: Float,
{
    pub fn new(mean: F, weight: F) -> Self {
        Self { mean, weight }
    }
}

impl<F> Add for Centroid<F>
where
    F: Float,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let new_weight = self.weight + other.weight;
        Centroid {
            mean: ((self.mean * self.weight) + (other.mean * other.weight)) / new_weight,
            weight: new_weight,
        }
    }
}

impl<F> Add for &Centroid<F>
where
    F: Float,
{
    type Output = Centroid<F>;

    fn add(self, other: &Centroid<F>) -> Centroid<F> {
        let new_weight = self.weight + other.weight;
        Centroid {
            mean: ((self.mean * self.weight) + (other.mean * other.weight)) / new_weight,
            weight: new_weight,
        }
    }
}

impl<F> Default for Centroid<F>
where
    F: Float,
{
    fn default() -> Self {
        Self::new(F::from(0.0).unwrap(), F::from(0.0).unwrap())
    }
}
