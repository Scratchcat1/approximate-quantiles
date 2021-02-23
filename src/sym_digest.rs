use crate::traits::{Digest, OwnedSize};
use num_traits::Float;
use std::cmp::Ordering;

/// Digest wrapper combining two asymmetrically accurate digests to
/// form a digest which has the same accuracy on both sides.
#[derive(Debug, Clone)]
pub struct SymDigest<D> {
    /// Lower part digest
    pub low_digest: D,
    /// Higher part digest
    pub high_digest: D,
}

impl<D> OwnedSize for SymDigest<D> {
    fn owned_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl<D, F> Digest<F> for SymDigest<D>
where
    D: Digest<F>,
    F: Float,
{
    fn add(&mut self, item: F) {
        self.low_digest.add(item);
        self.high_digest.add(-item);
    }

    fn add_buffer(&mut self, items: &[F]) {
        self.low_digest.add_buffer(items);
        self.high_digest
            .add_buffer(&items.iter().map(|x| -*x).collect::<Vec<F>>());
    }

    /// Additional information
    /// May behave strangely when estimating at the median where it
    /// will average between the two estimates.
    fn est_quantile_at_value(&mut self, value: F) -> F {
        let low_digest_est = self.low_digest.est_quantile_at_value(value);
        let high_digest_est =
            F::from(1.0).unwrap() - self.high_digest.est_quantile_at_value(-value);
        let avg = (low_digest_est + high_digest_est) / F::from(2.0).unwrap();

        // Compare the average quantile estimate to the median quantile to determine which
        // digest is likely to be more accurate
        match avg.partial_cmp(&F::from(0.5).unwrap()).unwrap() {
            Ordering::Less => low_digest_est,
            Ordering::Greater => high_digest_est,
            Ordering::Equal => avg,
        }
    }

    fn est_value_at_quantile(&mut self, target_quantile: F) -> F {
        let low_est = self.low_digest.est_value_at_quantile(target_quantile);
        let high_est = -self
            .high_digest
            .est_value_at_quantile(F::from(1.0).unwrap() - target_quantile);

        // Use the digest with which the estimate will be more accurate.
        match target_quantile.partial_cmp(&F::from(0.5).unwrap()).unwrap() {
            Ordering::Less => low_est,
            Ordering::Greater => high_est,
            Ordering::Equal => (low_est + high_est) / F::from(2.0).unwrap(),
        }
    }

    fn count(&self) -> u64 {
        return self.low_digest.count();
    }
}

impl<D> SymDigest<D> {
    pub fn new(low_digest: D, high_digest: D) -> Self {
        SymDigest {
            low_digest,
            high_digest,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::rc_sketch::rc_sketch::RCSketch;
    use crate::sym_digest::SymDigest;
    use crate::traits::Digest;
    use crate::util::gen_asc_vec;
    use crate::util::linear_digest::LinearDigest;
    use approx::assert_relative_eq;
    use num_traits::Float;
    use rand::distributions::{Distribution, Uniform};

    fn new_symdigest<F>(input_size: usize, acc_param: usize) -> SymDigest<RCSketch<F>>
    where
        F: Float,
    {
        SymDigest::new(
            RCSketch::new(input_size, acc_param),
            RCSketch::new(input_size, acc_param),
        )
    }

    #[test]
    fn insert_single_value() {
        let mut sketch = new_symdigest(1024, 8);
        sketch.add(1.0);
        assert_relative_eq!(sketch.est_quantile_at_value(1.0), 0.5);
    }

    #[test]
    fn insert_multiple_values() {
        let mut sketch = new_symdigest(1024, 8);
        (0..1000).map(|x| sketch.add(x as f64)).for_each(drop);

        println!("{:?}", sketch);
        assert_relative_eq!(sketch.est_quantile_at_value(0.0), 0.001);
        assert_relative_eq!(sketch.est_quantile_at_value(1.0), 0.002);
        assert_relative_eq!(sketch.est_quantile_at_value(10.0), 0.011);
        assert_relative_eq!(sketch.est_quantile_at_value(500.0), 0.5, epsilon = 10.0);
        assert_relative_eq!(sketch.est_quantile_at_value(1000.0), 1.0, epsilon = 30.0);
    }

    #[test]
    fn insert_descending_multiple_values() {
        let mut sketch = new_symdigest(1024, 8);
        (0..1000)
            .map(|x| sketch.add(999.0 - x as f64))
            .for_each(drop);

        println!("{:?}", sketch);
        assert_relative_eq!(sketch.est_quantile_at_value(0.0), 0.001);
        assert_relative_eq!(sketch.est_quantile_at_value(1.0), 0.002);
        assert_relative_eq!(sketch.est_quantile_at_value(500.0), 0.5, epsilon = 10.0);
        assert_relative_eq!(sketch.est_quantile_at_value(1000.0), 1.0, epsilon = 30.0);
        // assert_eq!(false, true);
    }

    #[test]
    fn insert_batch_single_value() {
        let mut sketch = new_symdigest(1024, 8);
        sketch.add_buffer(&vec![1.0]);
        assert_relative_eq!(sketch.est_quantile_at_value(1.0), 0.5);
    }

    #[test]
    fn insert_batch_multiple_values() {
        let mut sketch = new_symdigest(1024, 8);
        sketch.add_buffer(&gen_asc_vec(1000));

        println!("{:?}", sketch);
        assert_relative_eq!(sketch.est_quantile_at_value(0.0), 0.001);
        assert_relative_eq!(sketch.est_quantile_at_value(1.0), 0.002);
        assert_relative_eq!(sketch.est_quantile_at_value(10.0), 0.011);
        assert_relative_eq!(sketch.est_quantile_at_value(500.0), 0.5, epsilon = 10.0);
        assert_relative_eq!(sketch.est_quantile_at_value(1000.0), 1.0, epsilon = 30.0);
    }

    #[test]
    fn insert_batch_descending_multiple_values() {
        let mut sketch = new_symdigest(1024, 8);
        sketch.add_buffer(&(0..1000).map(|x| 999.0 - x as f64).collect::<Vec<f64>>());

        println!("{:?}", sketch);
        assert_relative_eq!(sketch.est_quantile_at_value(0.0), 0.001);
        assert_relative_eq!(sketch.est_quantile_at_value(1.0), 0.002);
        assert_relative_eq!(sketch.est_quantile_at_value(500.0), 0.5, epsilon = 10.0);
        assert_relative_eq!(sketch.est_quantile_at_value(1000.0), 1.0, epsilon = 30.0);
        // assert_eq!(false, true);
    }

    #[test]
    fn add_buffer_uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = new_symdigest(1_000_000, 200);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

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
    }

    #[test]
    fn add_buffer_uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = new_symdigest(1_000_000, 200);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

        println!("{:?}", digest);
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
    }

    #[test]
    fn est_value_at_quantile() {
        let mut sketch = new_symdigest(1024, 16);
        sketch.add_buffer(&gen_asc_vec(1000));

        println!("{:?}", sketch);
        assert_relative_eq!(sketch.est_value_at_quantile(0.0), 0.0, epsilon = 0.001);
        assert_relative_eq!(sketch.est_value_at_quantile(0.001), 1.0, epsilon = 0.1);
        assert_relative_eq!(sketch.est_value_at_quantile(0.1), 100.0, epsilon = 1.0);
        assert_relative_eq!(sketch.est_value_at_quantile(0.5), 500.0, epsilon = 4.0);
        assert_relative_eq!(sketch.est_value_at_quantile(1.0), 1000.0, epsilon = 10.0);
        // assert_eq!(false, true);
    }
}
