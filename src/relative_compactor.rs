use crate::traits::Digest;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct RCSketch {
    /// Vector of relative compactors
    pub buffers: Vec<Vec<f64>>,
    /// Upper bound on the number of inputs expected
    pub input_length: usize,
    /// Parameter controlling error and buffer size
    pub k: usize,
    /// Size of the buffers
    pub buffer_size: usize,
    /// Number of items seen
    pub count: u64,
    /// Compaction counter
    pub compaction_counter: u32,
}

impl Digest for RCSketch {
    fn add(&mut self, item: f64) {
        // Insert into the bottom buffer
        self.insert_at_rc(item, 0);
        self.count += 1;
    }

    fn add_buffer(&mut self, items: &[f64]) {
        let length = items.len() as u64;
        // Insert into the bottom buffer
        self.insert_at_rc_batch(items, 0);
        self.count += length;
    }

    fn est_quantile_at_value(&mut self, rank_item: f64) -> f64 {
        self.interpolate_rank(rank_item) as f64 / self.count as f64
    }

    fn est_value_at_quantile(&mut self, target_quantile: f64) -> f64 {
        let mut start = *self.buffers[0]
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let mut end = *self
            .buffers
            .last()
            .unwrap()
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let mut mid = (start + end) / 2.0;
        while (end - start).abs() > 0.00001 {
            mid = (start + end) / 2.0;
            let current_quantile = self.est_quantile_at_value(mid);

            match current_quantile.partial_cmp(&target_quantile).unwrap() {
                Ordering::Equal => return mid,
                Ordering::Less => start = mid,
                Ordering::Greater => end = mid,
            }
        }
        return mid;
    }
}

impl RCSketch {
    /// Create a new `RCSketch`
    /// # Arguments
    /// * `input_length` Upper bound on the number of inputs expected.
    /// * `k` Parameter controlling error and buffer size
    pub fn new(input_length: usize, k: usize) -> RCSketch {
        RCSketch {
            buffers: Vec::new(),
            input_length,
            k,
            buffer_size: Self::calc_buffer_size(input_length, k),
            count: 0,
            compaction_counter: 0,
        }
    }

    pub fn calc_buffer_size(input_length: usize, k: usize) -> usize {
        return usize::max(
            (2.0 * k as f64 * ((input_length / k) as f64).log2()) as usize,
            2 * k,
        );
    }

    /// Insert an item into a particular buffer in the sketch
    /// # Arguments
    /// * `item` The item to insert
    /// * `rc_index` the index of the buffer to insert at
    pub fn insert_at_rc(&mut self, item: f64, rc_index: usize) {
        // Create a new buffer if required
        if self.buffers.len() <= rc_index {
            self.buffers.push(Vec::with_capacity(self.buffer_size));
        }
        self.buffers[rc_index].push(item);
        // If buffer is full compact and insert into the next buffer
        if self.buffers[rc_index].len() >= self.buffer_size {
            let output_items = self.compact(rc_index);
            self.insert_at_rc_batch(&output_items, rc_index + 1);
        }
    }

    /// Insert multiple items into a particular buffer in the sketch
    /// # Arguments
    /// * `items` The items to insert
    /// * `rc_index` the index of the buffer to insert at
    pub fn insert_at_rc_batch(&mut self, items: &[f64], rc_index: usize) {
        // Create a new buffer if required
        if self.buffers.len() <= rc_index {
            self.buffers.push(Vec::with_capacity(self.buffer_size));
        }
        self.buffers[rc_index].extend(items);
        // If buffer is full compact and insert into the next buffer
        while self.buffers[rc_index].len() >= self.buffer_size {
            let output_items = self.compact(rc_index);
            self.insert_at_rc_batch(&output_items, rc_index + 1);
        }
    }

    /// Compact a particular buffer and return the output
    /// # Arguments
    /// * `rc_index` Index of the buffer to compact
    pub fn compact(&mut self, rc_index: usize) -> Vec<f64> {
        // Determine the index to remove after
        let compact_index = self.buffers[rc_index].len()
            - (self.compaction_counter.trailing_ones() as usize + 1) * self.k;
        self.compaction_counter += 1;
        // Sort and extract the largest values
        self.buffers[rc_index].sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let upper = self.buffers[rc_index].split_off(compact_index);
        // Remove half the largest values
        upper
            .into_iter()
            .enumerate()
            .filter(|(pos, _value)| pos % 2 == 0)
            .map(|(_pos, value)| value)
            .collect()
    }

    /// Estimate the rank of an item in the input set of the sketch
    /// # Arguments
    /// * `rank_item` Item to estimate the rank of
    pub fn interpolate_rank(&self, rank_item: f64) -> u64 {
        let mut rank = 0;
        for i in 0..self.buffers.len() {
            rank += self.buffers[i].iter().filter(|x| **x <= rank_item).count() as u64 * (1 << i);
        }
        rank
    }
}

#[cfg(test)]
mod test {
    use crate::relative_compactor::RCSketch;
    use crate::traits::Digest;
    use crate::util::gen_asc_vec;
    use crate::util::linear_digest::LinearDigest;
    use approx::assert_relative_eq;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn insert_single_value() {
        let mut sketch = RCSketch::new(1024, 8);
        sketch.add(1.0);
        assert_eq!(sketch.interpolate_rank(1.0), 1);
    }

    #[test]
    fn insert_multiple_values() {
        let mut sketch = RCSketch::new(1024, 8);
        (0..1000).map(|x| sketch.add(x as f64)).for_each(drop);

        println!("{:?}", sketch);
        assert_eq!(sketch.interpolate_rank(0.0), 1);
        assert_eq!(sketch.interpolate_rank(1.0), 2);
        assert_eq!(sketch.interpolate_rank(10.0), 11);
        assert_relative_eq!(
            sketch.interpolate_rank(500.0) as f64,
            500 as f64,
            epsilon = 10.0
        );
        assert_relative_eq!(
            sketch.interpolate_rank(1000.0) as f64,
            1000 as f64,
            epsilon = 30.0
        );
    }

    #[test]
    fn insert_descending_multiple_values() {
        let mut sketch = RCSketch::new(1024, 8);
        (0..1000)
            .map(|x| sketch.add(999.0 - x as f64))
            .for_each(drop);

        println!("{:?}", sketch);
        assert_eq!(sketch.interpolate_rank(0.0), 1);
        assert_eq!(sketch.interpolate_rank(1.0), 2);
        assert_relative_eq!(
            sketch.interpolate_rank(500.0) as f64,
            500 as f64,
            epsilon = 10.0
        );
        assert_relative_eq!(
            sketch.interpolate_rank(1000.0) as f64,
            1000 as f64,
            epsilon = 30.0
        );
        // assert_eq!(false, true);
    }

    #[test]
    fn insert_batch_single_value() {
        let mut sketch = RCSketch::new(1024, 8);
        sketch.add_buffer(&vec![1.0]);
        assert_eq!(sketch.interpolate_rank(1.0), 1);
    }

    #[test]
    fn insert_batch_multiple_values() {
        let mut sketch = RCSketch::new(1024, 8);
        sketch.add_buffer(&gen_asc_vec(1000));

        println!("{:?}", sketch);
        assert_eq!(sketch.interpolate_rank(0.0), 1);
        assert_eq!(sketch.interpolate_rank(1.0), 2);
        assert_eq!(sketch.interpolate_rank(10.0), 11);
        assert_relative_eq!(
            sketch.interpolate_rank(500.0) as f64,
            500 as f64,
            epsilon = 10.0
        );
        assert_relative_eq!(
            sketch.interpolate_rank(1000.0) as f64,
            1000 as f64,
            epsilon = 30.0
        );
    }

    #[test]
    fn insert_batch_descending_multiple_values() {
        let mut sketch = RCSketch::new(1024, 8);
        sketch.add_buffer(&(0..1000).map(|x| 999.0 - x as f64).collect::<Vec<f64>>());

        println!("{:?}", sketch);
        assert_eq!(sketch.interpolate_rank(0.0), 1);
        assert_eq!(sketch.interpolate_rank(1.0), 2);
        assert_relative_eq!(
            sketch.interpolate_rank(500.0) as f64,
            500 as f64,
            epsilon = 10.0
        );
        assert_relative_eq!(
            sketch.interpolate_rank(1000.0) as f64,
            1000 as f64,
            epsilon = 30.0
        );
        // assert_eq!(false, true);
    }

    #[test]
    fn add_buffer_uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = RCSketch::new(1_000_000, 10);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

        // assert_relative_eq!(
        //     digest.est_value_at_quantile(0.0) / linear_digest.est_value_at_quantile(0.0),
        //     1.0,
        //     epsilon = 0.01
        // );
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
        let mut digest = RCSketch::new(1_000_000, 10);
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
    }

    #[test]
    fn est_value_at_quantile() {
        let mut sketch = RCSketch::new(1024, 8);
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
