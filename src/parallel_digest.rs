use crate::traits::{Digest, OwnedSize};
use num_traits::Float;
use rayon::prelude::*;
use std::cmp::Ordering;

pub struct ParallelDigest<D, F>
where
    F: Float,
    D: Digest<F> + Send + Sync,
{
    pub digests: Vec<D>,
    pub min: F,
    pub max: F,
}

impl<D, F> Digest<F> for ParallelDigest<D, F>
where
    F: Float + Sync + Send,
    D: Digest<F> + Send + Sync,
{
    fn add(&mut self, item: F) {
        self.add_buffer(&[item]);
    }

    fn add_buffer(&mut self, buffer: &[F]) {
        self.min = F::min(
            self.min,
            *buffer
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );
        self.max = F::max(
            self.max,
            *buffer
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );
        let chunks = buffer.par_chunks((buffer.len() / self.digests.len()).max(1));
        chunks
            .zip(self.digests.par_iter_mut())
            .for_each(|(chunk, d)| d.add_buffer(chunk));
    }

    fn est_quantile_at_value(&mut self, value: F) -> F {
        let est_rank: F = self
            .digests
            .par_iter_mut()
            .map(|d| d.est_quantile_at_value(value) * F::from(d.count()).unwrap())
            .reduce(|| F::from(0.0).unwrap(), |acc, x| acc + x);
        est_rank / F::from(self.count()).unwrap()
    }

    fn est_value_at_quantile(&mut self, target_quantile: F) -> F {
        let mut start = self.min;
        let mut end = self.max;
        let mut mid = (start + end) / F::from(2.0).unwrap();
        // Max and min must be used as if start or end == 0 the proportional difference remains around 1.
        while (end - start).abs() / (self.min.abs() + self.max.abs()) > F::from(1e-6).unwrap() {
            mid = (start + end) / F::from(2.0).unwrap();
            let current_quantile = self.est_quantile_at_value(mid);

            // Don't return immediately on a match, this avoids high errors when looking for very small quantiles
            // Example [-100, -4, -3, -2, 0]
            // First round: mid = -50 and would satisfy q = 0.25 and return -50 instead of -4
            match current_quantile.partial_cmp(&target_quantile).unwrap() {
                Ordering::Equal => start = mid,
                Ordering::Less => start = mid,
                Ordering::Greater => end = mid,
            }
        }

        // Pick the smallest of the quantiles which is greater than or equal to the target quantile
        if self.est_quantile_at_value(start) >= target_quantile {
            return start;
        } else if self.est_quantile_at_value(mid) >= target_quantile {
            return mid;
        } else {
            return end;
        }
    }

    fn count(&self) -> u64 {
        self.digests.iter().map(|d| d.count()).sum()
    }
}

impl<D, F> ParallelDigest<D, F>
where
    F: Float,
    D: Digest<F> + Send + Sync,
{
    pub fn new(digests: Vec<D>) -> Self {
        Self {
            digests,
            min: F::max_value(),
            max: F::min_value(),
        }
    }
}

impl<D, F> OwnedSize for ParallelDigest<D, F>
where
    F: Float,
    D: Digest<F> + Send + Sync + OwnedSize,
{
    fn owned_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<D>() * self.digests.capacity()
            + self.digests.iter().map(|d| d.owned_size()).sum::<usize>()
    }
}

#[cfg(test)]
mod test {
    use crate::parallel_digest::ParallelDigest;
    use crate::traits::Digest;
    use crate::util::gen_asc_vec;
    use crate::util::linear_digest::LinearDigest;
    use approx::assert_relative_eq;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn add_buffer_with_many_centroids() {
        let buffer = gen_asc_vec(1001);
        let mut digest = ParallelDigest::new(vec![
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
        ]);
        digest.add_buffer(&buffer);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 500.0, epsilon = 2.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0, epsilon = 1.0);
    }

    #[test]
    fn add_buffer_uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = ParallelDigest::new(vec![
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
        ]);

        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

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
            epsilon = 0.0075
        );
        assert_eq!(digest.count(), linear_digest.values.len() as u64);
    }

    #[test]
    fn add_buffer_uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = ParallelDigest::new(vec![
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
        ]);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

        assert_relative_eq!(
            digest.est_quantile_at_value(0.0),
            linear_digest.est_quantile_at_value(0.0)
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(1.0) / linear_digest.est_quantile_at_value(1.0),
            1.0,
            epsilon = 0.005
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
        assert_eq!(digest.count(), linear_digest.values.len() as u64);
    }

    #[test]
    fn est_quantile_at_value() {
        let buffer: Vec<f64> = (0..1000).map(|x| -500.0 + x as f64).collect();
        let mut digest = ParallelDigest::new(vec![
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
        ]);
        digest.add_buffer(&buffer);

        let mut linear_digest = LinearDigest::new();
        linear_digest.add_buffer(&buffer);

        assert_relative_eq!(
            digest.est_quantile_at_value(-500.0),
            linear_digest.est_quantile_at_value(-500.0),
            epsilon = 0.0005
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(-250.0) / linear_digest.est_quantile_at_value(-250.0),
            1.0,
            epsilon = 0.0025
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(0.0) / linear_digest.est_quantile_at_value(0.0),
            1.0,
            epsilon = 0.001
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(250.0) / linear_digest.est_quantile_at_value(250.0),
            1.0,
            epsilon = 0.001
        );
        assert_relative_eq!(
            digest.est_quantile_at_value(500.0) / linear_digest.est_quantile_at_value(500.0),
            1.0,
            epsilon = 0.0005
        );
    }

    #[test]
    fn est_value_at_quantile_singleton_centroids() {
        let mut digest = ParallelDigest::new(vec![
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
            LinearDigest::new(),
        ]);
        digest.add_buffer(&vec![1.0, 2.0, 8.0, 0.5]);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.5);
        assert_relative_eq!(digest.est_value_at_quantile(0.24), 0.5, epsilon = 1e-5);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 1.0, epsilon = 1e-5);
        assert_relative_eq!(digest.est_value_at_quantile(0.49), 1.0, epsilon = 1e-5);
        assert_relative_eq!(digest.est_value_at_quantile(0.50), 2.0, epsilon = 1e-5);
        assert_relative_eq!(digest.est_value_at_quantile(0.74), 2.0, epsilon = 1e-5);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 8.0, epsilon = 1e-5);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 8.0);
        assert_eq!(digest.count(), 4);
    }
}
