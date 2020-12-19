use crate::traits::Digest;
use num_traits::{cast::ToPrimitive, Float};

pub struct LinearDigest<F>
where
    F: Float + ToPrimitive,
{
    pub values: Vec<F>,
}

impl<F> LinearDigest<F>
where
    F: Float + ToPrimitive,
{
    pub fn new() -> Self {
        LinearDigest { values: Vec::new() }
    }
}

impl<F> Digest<F> for LinearDigest<F>
where
    F: Float + ToPrimitive,
{
    fn add(&mut self, item: F) {
        self.values.push(item);
    }

    fn add_buffer(&mut self, items: &[F]) {
        self.values.extend(items);
    }

    fn est_quantile_at_value(&mut self, target_value: F) -> F {
        let less_than = F::from(self.values.iter().filter(|x| **x < target_value).count()).unwrap();
        let equal_to = F::from(self.values.iter().filter(|x| **x == target_value).count()).unwrap();
        if equal_to <= F::from(1.0).unwrap() {
            // If the one or zero such values exist there are not values to the left of the midpoint of values equal to the target value.
            less_than / F::from(self.values.len()).unwrap()
        } else {
            (less_than + equal_to / F::from(2.0).unwrap()) / F::from(self.values.len()).unwrap()
        }
    }

    fn est_value_at_quantile(&mut self, target_quantile: F) -> F {
        let f_len = F::from(self.values.len()).unwrap();
        let target_index = (target_quantile * f_len).round().to_usize().unwrap();
        self.values.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        return if target_index == self.values.len() {
            *self.values.last().unwrap()
        } else {
            self.values[target_index]
        };
    }
}

#[cfg(test)]
mod test {
    use crate::traits::Digest;
    use crate::util::linear_digest::LinearDigest;
    use approx::assert_relative_eq;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn est_quantile_at_value() {
        let dataset: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let mut digest = LinearDigest::new();
        digest.add_buffer(&dataset);

        assert_relative_eq!(digest.est_quantile_at_value(0.0), 0.0);
        assert_relative_eq!(digest.est_quantile_at_value(250.0), 0.25);
        assert_relative_eq!(digest.est_quantile_at_value(500.0), 0.5);
        assert_relative_eq!(digest.est_quantile_at_value(750.0), 0.75);
        assert_relative_eq!(digest.est_quantile_at_value(1000.0), 1.0);
    }

    #[test]
    fn uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0..1000);
        let dataset: Vec<f64> = (0..100_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();

        // dataset.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut digest = LinearDigest::new();
        digest.add_buffer(&dataset);

        assert_relative_eq!(
            digest.est_quantile_at_value(0.0),
            0.5 * dataset.iter().filter(|x| **x == 0.0).count() as f64 / dataset.len() as f64
        );
        assert_relative_eq!(digest.est_quantile_at_value(250.0), 0.25, epsilon = 0.01);
        assert_relative_eq!(digest.est_quantile_at_value(500.0), 0.5, epsilon = 0.01);
        assert_relative_eq!(digest.est_quantile_at_value(750.0), 0.75, epsilon = 0.01);
        assert_relative_eq!(digest.est_quantile_at_value(999.0), 1.0, epsilon = 0.01);
    }

    #[test]
    fn est_value_at_quantile() {
        let dataset: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let mut digest = LinearDigest::new();
        digest.add_buffer(&dataset);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 999.0);
    }

    #[test]
    fn uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0..1001);
        let dataset: Vec<f64> = (0..100_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();

        let mut digest = LinearDigest::new();
        digest.add_buffer(&dataset);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0, epsilon = 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 3.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0, epsilon = 4.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 3.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0, epsilon = 2.0);
    }
}
