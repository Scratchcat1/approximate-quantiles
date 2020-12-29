use crate::traits::Digest;
use num_traits::{Float, NumAssignOps};

pub struct MultiDigest<F>
where
    F: Float,
{
    pub digests: Vec<Box<dyn Digest<F>>>,
}

impl<F> MultiDigest<F>
where
    F: Float,
{
    pub fn new() -> Self {
        MultiDigest {
            digests: Vec::new(),
        }
    }

    pub fn add_digest(&mut self, digest: Box<dyn Digest<F>>) {
        self.digests.push(digest);
    }
}

impl<F> Digest<F> for MultiDigest<F>
where
    F: Float + NumAssignOps,
{
    fn add(&mut self, item: F) {
        for digest in &mut self.digests {
            digest.add(item);
        }
    }

    fn add_buffer(&mut self, items: &[F]) {
        for digest in &mut self.digests {
            digest.add_buffer(items);
        }
    }

    fn est_quantile_at_value(&mut self, value: F) -> F {
        let mut sum = F::from(0.0).unwrap();
        for digest in &mut self.digests {
            sum += digest.est_quantile_at_value(value);
        }
        sum / F::from(self.digests.len()).unwrap()
    }

    fn est_value_at_quantile(&mut self, target_quantile: F) -> F {
        let mut sum = F::from(0.0).unwrap();
        for digest in &mut self.digests {
            sum += digest.est_value_at_quantile(target_quantile);
        }
        sum / F::from(self.digests.len()).unwrap()
    }
}
