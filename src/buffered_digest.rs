use crate::traits::{Digest, OwnedSize};
use num_traits::Float;

#[derive(Clone)]
pub struct BufferedDigest<T, F>
where
    T: Digest<F>,
    F: Float,
{
    digest: T,
    buffer: Vec<F>,
    capacity: usize,
}

impl<T, F> OwnedSize for BufferedDigest<T, F>
where
    T: Digest<F> + OwnedSize,
    F: Float,
{
    fn owned_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<F>() * self.buffer.len()
            + (self.digest.owned_size() - std::mem::size_of::<T>()) // Don't count the size of the digest twice.
    }
}

impl<T, F> BufferedDigest<T, F>
where
    T: Digest<F>,
    F: Float,
{
    pub fn new(digest: T, capacity: usize) -> BufferedDigest<T, F> {
        BufferedDigest {
            digest,
            buffer: Vec::new(),
            capacity,
        }
    }

    pub fn flush(&mut self) {
        self.digest.add_buffer(&self.buffer);
        self.buffer.clear();
    }
}

impl<T, F> Digest<F> for BufferedDigest<T, F>
where
    T: Digest<F>,
    F: Float,
{
    fn add(&mut self, item: F) {
        self.buffer.push(item);
        if self.buffer.len() > self.capacity {
            self.flush();
        }
    }

    fn add_buffer(&mut self, items: &[F]) {
        items.chunks(self.capacity).for_each(|chunk| {
            self.buffer.extend(chunk);
            if self.buffer.len() > self.capacity {
                self.flush();
            }
        })
    }

    fn est_quantile_at_value(&mut self, value: F) -> F {
        self.flush();
        self.digest.est_quantile_at_value(value)
    }

    fn est_value_at_quantile(&mut self, quantile: F) -> F {
        self.flush();
        self.digest.est_value_at_quantile(quantile)
    }
}
