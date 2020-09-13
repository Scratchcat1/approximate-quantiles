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
