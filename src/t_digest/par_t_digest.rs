use crate::t_digest::centroid::Centroid;
use crate::t_digest::t_digest::TDigest;
use crate::traits::Digest;
use rayon::prelude::*;

pub struct ParTDigest<C, F, G>
where
    F: Fn(f64, f64) -> f64 + Sync,
    G: Fn(f64, f64) -> f64 + Sync,
    C: Fn() -> TDigest<F, G> + Sync,
{
    pub digest: TDigest<F, G>,
    threads: usize,
    pub buffer: Vec<f64>,
    capacity: usize,
    creator: C,
}

impl<C, F, G> ParTDigest<C, F, G>
where
    F: Fn(f64, f64) -> f64 + Sync,
    G: Fn(f64, f64) -> f64 + Sync,
    C: Fn() -> TDigest<F, G> + Sync,
{
    pub fn new(threads: usize, capacity: usize, creator: C) -> ParTDigest<C, F, G> {
        ParTDigest {
            digest: creator(),
            threads,
            buffer: Vec::new(),
            capacity,
            creator,
        }
    }

    pub fn flush(&mut self) {
        let centroids: Vec<Centroid> = self
            .buffer
            .par_chunks(self.threads)
            .map(|chunk| {
                let mut tmp_digest = (self.creator)();
                tmp_digest.add_buffer(chunk.to_vec());
                tmp_digest.centroids
            })
            .flatten()
            .collect();
        self.digest.add_centroid_buffer(centroids);
        self.buffer.clear();
    }

    pub fn total_weight(&self) -> f64 {
        self.digest.total_weight()
    }
}

impl<C, F, G> Digest for ParTDigest<C, F, G>
where
    F: Fn(f64, f64) -> f64 + Sync + Sync,
    G: Fn(f64, f64) -> f64 + Sync + Sync,
    C: Fn() -> TDigest<F, G> + Sync,
{
    fn add(&mut self, item: f64) {
        self.buffer.push(item);
        if self.buffer.len() > self.capacity {
            self.flush();
        }
    }
    fn add_buffer(&mut self, items: Vec<f64>) {
        self.buffer.extend(items);
        // self.buffer = items;
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

#[cfg(test)]
mod test {
    use crate::t_digest::par_t_digest::ParTDigest;
    use crate::t_digest::scale_functions::{inv_k1, k1};
    use crate::t_digest::t_digest::TDigest;
    use crate::traits::Digest;
    use approx::assert_relative_eq;

    #[test]
    fn add_buffer_with_multiple_centroid() {
        let buffer = vec![1.0, 2.0, 0.5];
        let mut digest = ParTDigest::new(32, 128, &|| TDigest::new(&k1, &inv_k1, 50.0));
        digest.add_buffer(buffer);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.5);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 0.625);
        assert_relative_eq!(digest.est_value_at_quantile(0.5), 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 1.75);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 2.0);
        assert_eq!(digest.total_weight(), 3.0);
    }

    #[test]
    fn add_buffer_with_many_centroids() {
        let buffer = (0..1001).map(|x| x as f64).collect();
        let mut digest = ParTDigest::new(250, 1000, &|| TDigest::new(&k1, &inv_k1, 50.0));
        digest.add_buffer(buffer);

        println!("{:?}", digest.digest.centroids);
        println!("{:?}", digest.digest.min);
        println!("{:?}", digest.digest.max);
        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(
            digest.est_value_at_quantile(0.5),
            500.0,
            epsilon = 0.0000001
        );
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }

    #[test]
    fn add_buffer_with_many_centroids_high_compression() {
        let buffer = (0..1001).map(|x| x as f64).collect();
        let mut digest = ParTDigest::new(32, 128, &|| TDigest::new(&k1, &inv_k1, 20.0));
        digest.add_buffer(buffer);

        assert_relative_eq!(digest.est_value_at_quantile(0.0), 0.0, epsilon = 3.0);
        assert_relative_eq!(digest.est_value_at_quantile(0.25), 250.0, epsilon = 1.0);
        assert_relative_eq!(
            digest.est_value_at_quantile(0.5),
            500.0,
            epsilon = 0.0000001
        );
        assert_relative_eq!(digest.est_value_at_quantile(0.75), 750.0, epsilon = 1.0);
        assert_relative_eq!(digest.est_value_at_quantile(1.0), 1000.0);
        assert_eq!(digest.total_weight(), 1001.0);
    }

    #[test]
    fn est_quantile_at_value() {
        let buffer = (0..1001).map(|x| -500.0 + x as f64).collect();
        let mut digest = ParTDigest::new(32, 128, &|| TDigest::new(&k1, &inv_k1, 20.0));
        digest.add_buffer(buffer);

        assert_relative_eq!(digest.est_quantile_at_value(-500.0), 0.0);
        assert_relative_eq!(digest.est_quantile_at_value(-250.0), 0.25, epsilon = 0.001);
        assert_relative_eq!(digest.est_quantile_at_value(0.0), 0.5, epsilon = 0.001);
        assert_relative_eq!(digest.est_quantile_at_value(250.0), 0.75, epsilon = 0.001);
        assert_relative_eq!(digest.est_quantile_at_value(500.0), 1.0);
    }
}
