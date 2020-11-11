use crate::traits::Digest;

pub struct BufferedDigest<T>
where
    T: Digest,
{
    digest: T,
    buffer: Vec<f64>,
    capacity: usize,
}

impl<T> BufferedDigest<T>
where
    T: Digest,
{
    pub fn new(digest: T, capacity: usize) -> BufferedDigest<T> {
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

impl<T> Digest for BufferedDigest<T>
where
    T: Digest,
{
    fn add(&mut self, item: f64) {
        self.buffer.push(item);
        if self.buffer.len() > self.capacity {
            self.flush();
        }
    }

    fn add_buffer(&mut self, items: &[f64]) {
        self.buffer.extend(items);
        if self.buffer.len() > self.capacity {
            self.flush();
        }
    }

    fn est_quantile_at_value(&mut self, value: f64) -> f64 {
        self.flush();
        self.digest.est_quantile_at_value(value)
    }

    fn est_value_at_quantile(&mut self, quantile: f64) -> f64 {
        self.flush();
        self.digest.est_value_at_quantile(quantile)
    }
}
