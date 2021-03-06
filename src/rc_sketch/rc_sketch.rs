use crate::rc_sketch::compaction_method::CompactionMethod;
use crate::t_digest::centroid::Centroid;
use crate::traits::{Digest, OwnedSize};
use num_traits::{cast::ToPrimitive, Float, NumAssignOps};
use rand_distr::{Distribution, Uniform};

#[derive(Debug, Clone)]
pub struct RCSketch<F>
where
    F: Float,
{
    /// Vector of relative compactors
    pub buffers: Vec<Vec<F>>,
    /// Upper bound on the number of inputs expected
    pub input_length: usize,
    /// Parameter controlling error and buffer size
    pub k: usize,
    /// Size of the buffers
    pub buffer_size: usize,
    /// Number of items seen
    pub count: u64,
    /// Compaction counter
    pub compaction_counters: Vec<u32>,
}

impl<F> OwnedSize for RCSketch<F>
where
    F: Float,
{
    fn owned_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<Vec<F>>() * self.buffers.capacity()
            + self
                .buffers
                .iter()
                .map(|buffer| std::mem::size_of::<F>() * buffer.capacity())
                .sum::<usize>()
            + std::mem::size_of::<u32>() * self.compaction_counters.capacity()
    }
}

impl<F> Digest<F> for RCSketch<F>
where
    F: Float + ToPrimitive + NumAssignOps,
{
    fn add(&mut self, item: F) {
        // Insert into the bottom buffer
        self.insert_at_rc(item, 0, false, CompactionMethod::Default);
        self.count += 1;
    }

    fn add_buffer(&mut self, items: &[F]) {
        let length = items.len() as u64;
        // Insert into the bottom buffer
        items
            .chunks(self.buffer_size / 2)
            .for_each(|chunk| self.insert_at_rc_batch(chunk, 0, false, CompactionMethod::Default));
        self.count += length;
    }

    fn est_quantile_at_value(&mut self, rank_item: F) -> F {
        F::from(self.interpolate_rank(rank_item)).unwrap() / F::from(self.count).unwrap()
    }

    fn est_value_at_quantile(&mut self, target_quantile: F) -> F {
        let centroids = self.sorted_weighted_values();
        let mut total_weight = F::from(0.0).unwrap();

        for centroid in &centroids {
            if centroid.weight + total_weight > target_quantile * F::from(self.count).unwrap() {
                return centroid.mean;
            }
            total_weight += centroid.weight;
        }
        centroids.last().expect("Sketch is empty").mean
    }

    fn count(&self) -> u64 {
        return self.count;
    }
}

impl<F> RCSketch<F>
where
    F: Float + ToPrimitive,
{
    /// Create a new `RCSketch`
    /// # Arguments
    /// * `input_length` Upper bound on the number of inputs expected.
    /// * `k` Parameter controlling error and buffer size
    pub fn new(input_length: usize, k: usize) -> RCSketch<F> {
        RCSketch {
            buffers: Vec::new(),
            input_length,
            k,
            buffer_size: Self::calc_buffer_size(input_length, k),
            count: 0,
            compaction_counters: Vec::new(),
        }
    }

    pub fn calc_buffer_size(input_length: usize, k: usize) -> usize {
        return usize::max(
            (F::from(2.0).unwrap()
                * F::from(k).unwrap()
                * F::from(input_length / k).unwrap().log2().ceil())
            .to_usize()
            .unwrap_or(0),
            2 * k,
        );
    }

    /// Determine the compaction index using an exponential distribution
    /// Should only be called once per compaction as the compaction counter is updated.
    /// # Arguments
    /// * `rc_index` Index of the buffer to determine the compaction index of
    pub fn get_compact_index(&mut self, rc_index: usize) -> usize {
        // Determine the index to remove after
        let compact_index = self.buffers[rc_index].len()
            - (self.compaction_counters[rc_index].trailing_ones() as usize + 1) * self.k;
        self.compaction_counters[rc_index] += 1;
        return compact_index;
    }

    /// Determine the compaction index for a buffer.
    /// Index is always half the target buffer size
    /// # Arguments
    /// * `rc_index` Index of the buffer to determine the compaction index of
    #[inline]
    pub fn get_compact_index_fast(&self) -> usize {
        return self.buffer_size / 2;
    }

    pub fn add_buffer_fast(&mut self, items: &[F]) {
        self.add_buffer_custom(items, true, CompactionMethod::Default)
    }

    /// Insert items into in the sketch
    /// * `fast_compaction` Enables faster compaction in exchange for potentially increased error
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn add_buffer_custom(
        &mut self,
        items: &[F],
        fast_compaction: bool,
        compaction_method: CompactionMethod,
    ) {
        let length = items.len() as u64;
        // Insert into the bottom buffer
        items.chunks(self.buffer_size / 2).for_each(|chunk| {
            self.insert_at_rc_batch(chunk, 0, fast_compaction, compaction_method)
        });
        self.count += length;
    }

    /// Insert an item into a particular buffer in the sketch
    /// # Arguments
    /// * `item` The item to insert
    /// * `rc_index` The index of the buffer to insert at
    /// * `fast_compaction` Enables faster compaction in exchange for potentially increased error
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn insert_at_rc(
        &mut self,
        item: F,
        rc_index: usize,
        fast_compaction: bool,
        compaction_method: CompactionMethod,
    ) {
        // Create a new buffer if required
        if self.buffers.len() <= rc_index {
            self.buffers.push(Vec::with_capacity(self.buffer_size));
            self.compaction_counters.push(0);
        }
        self.buffers[rc_index].push(item);
        // If buffer is full compact and insert into the next buffer
        if self.buffers[rc_index].len() >= self.buffer_size {
            let compaction_index = if fast_compaction {
                self.get_compact_index_fast()
            } else {
                self.get_compact_index(rc_index)
            };
            let output_items = self.compact(rc_index, compaction_index, compaction_method);
            self.insert_at_rc_batch(
                &output_items,
                rc_index + 1,
                fast_compaction,
                compaction_method,
            );
        }
    }

    /// Insert multiple items into a particular buffer in the sketch
    /// # Arguments
    /// * `items` The items to insert
    /// * `rc_index` The index of the buffer to insert at
    /// * `fast_compaction` Enables faster compaction in exchange for potentially increased error
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn insert_at_rc_batch(
        &mut self,
        items: &[F],
        rc_index: usize,
        fast_compaction: bool,
        compaction_method: CompactionMethod,
    ) {
        // Create a new buffer if required
        if self.buffers.len() <= rc_index {
            self.buffers.push(Vec::with_capacity(self.buffer_size));
            self.compaction_counters.push(0);
        }

        // Copy the maximum number of items into the buffer each loop
        let mut current_index = 0;
        while current_index < items.len() {
            let end = usize::min(
                current_index + (self.buffer_size - self.buffers[rc_index].len()),
                items.len(),
            );
            self.buffers[rc_index].extend(&items[current_index..end]);
            current_index = end;
            // If buffer is full compact and insert into the next buffer
            // Buffer may be overfilled since more than one item was added so keep compacting until size is below the buffer size.
            while self.buffers[rc_index].len() >= self.buffer_size {
                let compaction_index = if fast_compaction {
                    self.get_compact_index_fast()
                } else {
                    self.get_compact_index(rc_index)
                };
                let output_items = self.compact(rc_index, compaction_index, compaction_method);
                self.insert_at_rc_batch(
                    &output_items,
                    rc_index + 1,
                    fast_compaction,
                    compaction_method,
                );
            }
        }
    }

    /// Compact a particular buffer and return the output
    /// # Arguments
    /// * `rc_index` Index of the buffer to compact
    /// * `compact_index` Index after which to compact
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn compact(
        &mut self,
        rc_index: usize,
        compact_index: usize,
        compaction_method: CompactionMethod,
    ) -> Vec<F> {
        // Sort and extract the largest values
        self.buffers[rc_index].sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let upper = self.buffers[rc_index].split_off(compact_index);

        match compaction_method {
            CompactionMethod::Default => {
                let mut rng = rand::thread_rng();
                let uniform = Uniform::new(0, 2);
                let chosen_pos = uniform.sample(&mut rng);
                // Remove half the largest values
                upper
                    .into_iter()
                    .enumerate()
                    .filter(|(pos, _value)| pos % 2 == chosen_pos)
                    .map(|(_pos, value)| value)
                    .collect()
            }
            CompactionMethod::AverageNeighbour => {
                let mut output = Vec::new();
                for i in 0..upper.len() / 2 {
                    output.push((upper[2 * i] + upper[2 * i + 1]) / F::from(2).unwrap());
                }
                output
            }
        }
    }

    /// Estimate the rank of an item in the input set of the sketch
    /// # Arguments
    /// * `rank_item` Item to estimate the rank of
    pub fn interpolate_rank(&self, rank_item: F) -> usize {
        let mut rank = 0;
        for i in 0..self.buffers.len() {
            rank += self.buffers[i].iter().filter(|x| **x <= rank_item).count() * (1 << i);
        }
        rank
    }

    /// Return sorted vector of centroids for each value in the digest
    /// Ascending order
    pub fn sorted_weighted_values(&self) -> Vec<Centroid<F>> {
        let mut centroids = vec![];
        for i in 0..self.buffers.len() {
            centroids.extend(self.buffers[i].iter().map(|x| Centroid {
                mean: *x,
                weight: F::from(1 << i).unwrap(),
            }));
        }
        centroids.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        centroids
    }
}

#[cfg(test)]
mod test {
    use crate::rc_sketch::rc_sketch::RCSketch;
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
        let mut digest = RCSketch::new(1_000_000, 200);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer(&buffer);
        linear_digest.add_buffer(&buffer);

        println!("{:?}", digest);
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
        let mut digest = RCSketch::new(1_000_000, 200);
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
    fn add_buffer_fast_uniform_est_value_at_quantile() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = RCSketch::new(1_000_000, 320);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer_fast(&buffer);
        linear_digest.add_buffer(&buffer);

        println!("{:?}", digest);
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
    fn add_buffer_fast_uniform_est_quantile_at_value() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(0.0..1001.0);
        let buffer: Vec<f64> = (0..1_000_000)
            .map(|_| uniform.sample(&mut rng) as f64)
            .collect();
        let mut digest = RCSketch::new(1_000_000, 315);
        let mut linear_digest = LinearDigest::new();
        digest.add_buffer_fast(&buffer);
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
        let mut sketch = RCSketch::new(1024, 16);
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
