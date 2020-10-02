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

/// Digest with internal buffer and parallel merging of that buffer.
/// Adds input items to an internal buffer.
/// When full the buffer is split into chunks, each of which are fed into
/// a new digest.
/// The result of each of these digests is fed into the main digest.
/// Buffer is flushed if an estimation is made.
impl<C, F, G> ParTDigest<C, F, G>
where
    F: Fn(f64, f64) -> f64 + Sync + Send,
    G: Fn(f64, f64) -> f64 + Sync + Send,
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
        if !self.buffer.is_empty() {
            let digests: Vec<TDigest<F, G>> = self
                .buffer
                .par_chunks(self.threads)
                .map(|chunk| {
                    let mut tmp_digest = (self.creator)();
                    tmp_digest.add_buffer(chunk.to_vec());
                    tmp_digest
                })
                .collect();
            self.digest.min = digests
                .iter()
                .min_by(|a, b| a.min.partial_cmp(&b.min).unwrap())
                .unwrap()
                .min;
            self.digest.max = digests
                .iter()
                .max_by(|a, b| a.max.partial_cmp(&b.max).unwrap())
                .unwrap()
                .max;
            self.digest.add_centroid_buffer(
                digests
                    .into_iter()
                    .map(|digest| digest.centroids)
                    .flatten()
                    .collect::<Vec<Centroid>>(),
            );
            self.buffer.clear();
        }
    }

    pub fn total_weight(&self) -> f64 {
        self.digest.total_weight()
    }
}

impl<C, F, G> Digest for ParTDigest<C, F, G>
where
    F: Fn(f64, f64) -> f64 + Sync + Send,
    G: Fn(f64, f64) -> f64 + Sync + Send,
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
    use crate::util::linear_digest::LinearDigest;
    use approx::assert_relative_eq;
    use rand::distributions::{Distribution, Uniform};

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
        let mut digest = ParTDigest::new(2500, 10000, &|| TDigest::new(&k1, &inv_k1, 50.0));
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
    fn add_buffer_uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = ParTDigest::new(2500, 10000, &|| TDigest::new(&k1, &inv_k1, 4000.0));
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(buffer.clone());
        linear_digest.add_buffer(buffer.clone());

        assert_relative_eq!(
            digest.est_value_at_quantile(0.0) / linear_digest.est_value_at_quantile(0.0),
            1.0,
            epsilon = 0.0005
        );
        assert_relative_eq!(
            digest.est_value_at_quantile(0.001) / linear_digest.est_value_at_quantile(0.001),
            1.0,
            epsilon = 0.005
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
            epsilon = 0.005
        );
        assert_eq!(digest.total_weight(), linear_digest.values.len() as f64);
    }

    #[test]
    fn add_buffer_uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = ParTDigest::new(250, 1000, &|| TDigest::new(&k1, &inv_k1, 2000.0));
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(buffer.clone());
        linear_digest.add_buffer(buffer.clone());

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
        assert_eq!(digest.total_weight(), linear_digest.values.len() as f64);
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
