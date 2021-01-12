use crate::traits::{Digest, OwnedSize};
use num_traits::Float;
use rayon::prelude::*;

pub struct ParallelDigest<F>
where
    F: Float,
{
    pub digests: Vec<Box<dyn Digest<F> + Send + Sync>>,
}

impl<F> Digest<F> for ParallelDigest<F>
where
    F: Float + Sync,
{
    fn add(&mut self, item: F) {}

    fn add_buffer(&mut self, buffer: &[F]) {
        let chunks = buffer.par_chunks(buffer.len() / self.digests.len());
        chunks
            .zip(self.digests.par_iter_mut())
            .for_each(|(chunk, d)| d.add_buffer(chunk));
    }

    fn est_quantile_at_value(&mut self, value: F) -> F {
        let est_rank: F = self
            .digests
            .iter_mut()
            .map(|d| d.est_quantile_at_value(value) * F::from(d.count()).unwrap())
            .fold(F::from(0.0).unwrap(), |acc, x| acc + x);
        est_rank / F::from(self.count()).unwrap()
    }

    fn est_value_at_quantile(&mut self, quantile: F) -> F {
        unimplemented!("");
    }

    fn count(&self) -> u64 {
        self.digests.iter().map(|d| d.count()).sum()
    }
}

impl<F> ParallelDigest<F>
where
    F: Float,
{
    pub fn new(digests: Vec<Box<dyn Digest<F> + Send + Sync>>) -> Self {
        Self { digests }
    }
}
