use crate::traits::Digest;

pub struct LinearDigest {
    pub values: Vec<f64>,
}

impl LinearDigest {
    pub fn new() -> Self {
        LinearDigest { values: Vec::new() }
    }
}

impl Digest for LinearDigest {
    fn add(&mut self, item: f64) {
        self.values.push(item);
    }

    fn add_buffer(&mut self, items: std::vec::Vec<f64>) {
        self.values.extend(items);
    }

    fn est_quantile_at_value(&mut self, target_value: f64) -> f64 {
        self.values.iter().filter(|x| **x < target_value).count() as f64 / self.values.len() as f64
    }

    fn est_value_at_quantile(&mut self, target_quantile: f64) -> f64 {
        let target_index = (target_quantile * self.values.len() as f64).round() as usize;
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
    fn uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0..1000);
        let dataset: Vec<f64> = (0..100_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();

        let mut digest = LinearDigest::new();
        digest.add_buffer(dataset);

        assert_relative_eq!(digest.est_quantile_at_value(0.0), 0.0);
        assert_relative_eq!(digest.est_quantile_at_value(250.0), 0.25, epsilon = 0.01);
        assert_relative_eq!(digest.est_quantile_at_value(500.0), 0.5, epsilon = 0.01);
        assert_relative_eq!(digest.est_quantile_at_value(750.0), 0.75, epsilon = 0.01);
        assert_relative_eq!(digest.est_quantile_at_value(1000.0), 1.0, epsilon = 0.01);
    }

    #[test]
    fn uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0..1001);
        let dataset: Vec<f64> = (0..100_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();

        let mut digest = LinearDigest::new();
        digest.add_buffer(dataset);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0, epsilon = 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 3.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0, epsilon = 3.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 3.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0, epsilon = 2.0);
    }
}
