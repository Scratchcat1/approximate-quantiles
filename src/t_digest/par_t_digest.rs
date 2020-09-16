use crate::t_digest::centroid::Centroid;
use crate::t_digest::t_digest::TDigest;
use crate::traits::Digest;
use rayon::prelude::*;

pub struct ParTDigest<'a> {
    pub digest: TDigest<'a>,
    threads: u32,
    pub buffer: Vec<f64>,
    capacity: usize,
    creator: &'a dyn Fn() -> TDigest<'a>,
}

impl<'a> ParTDigest<'_> {
    pub fn new(threads: u32, capacity: usize, creator: &'a dyn Fn() -> TDigest<'a>) -> ParTDigest {
        ParTDigest {
            digest: creator(),
            threads,
            buffer: Vec::new(),
            capacity,
            creator,
        }
    }

    pub fn flush(&mut self) {
        let elements: Vec<f64> = self.buffer.drain(0..self.buffer.len()).collect();
        let centroids: Vec<Centroid> = elements
            .chunks(128)
            .collect::<Vec<&[f64]>>()
            .par_iter()
            .map(|chunk| {
                let mut tmp_digest = (self.creator)();
                tmp_digest.add_buffer(chunk.to_vec());
                tmp_digest.centroids
            })
            .flatten()
            .collect();
        self.digest.add_centroid_buffer(centroids);
    }

    pub fn total_weight(&self) -> f64 {
        self.digest.total_weight()
    }
}

impl Digest for ParTDigest<'_> {
    fn add(&mut self, item: f64) {
        self.buffer.push(item);
        if self.buffer.len() > self.capacity {
            self.flush();
        }
    }
    fn add_buffer(&mut self, items: Vec<f64>) {
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
        let mut digest = ParTDigest::new(32, 128, &|| TDigest::new(&k1, &inv_k1, 50.0));
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
