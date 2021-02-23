use crate::rc_sketch::compaction_method::CompactionMethod;
use crate::t_digest::centroid::Centroid;
use crate::traits::{Digest, OwnedSize};
use num_traits::{cast::ToPrimitive, Float, NumAssignOps};
use rand_distr::{Distribution, Uniform};

#[derive(Debug, Clone)]
pub struct RCSketch2<F>
where
    F: Float,
{
    /// Vector of relative compactors
    pub buffers: Vec<Vec<F>>,
    /// Number of elements in the sketch
    pub sketch_size: usize,
    /// Parameter controlling error and buffer size
    pub k: usize,
    /// Number of items seen
    pub count: u64,
    /// Compaction counter
    pub compaction_counters: Vec<u32>,
    /// Num sections
    pub number_of_sections: Vec<u32>,
    /// Section sizes in float form
    pub section_sizes: Vec<f32>,
}

impl<F> OwnedSize for RCSketch2<F>
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
            + self.compaction_counters.capacity()
                * (std::mem::size_of::<u32>() * 3 + std::mem::size_of::<f32>())
    }
}

impl<F> Digest<F> for RCSketch2<F>
where
    F: Float + NumAssignOps,
{
    fn add(&mut self, item: F) {
        // Insert into the bottom buffer
        self.insert(item, false, CompactionMethod::Default);
        self.count += 1;
    }

    fn add_buffer(&mut self, items: &[F]) {
        let length = items.len() as u64;
        // Insert into the bottom buffer
        self.insert_batch(items, false, CompactionMethod::Default);
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

impl<F> RCSketch2<F>
where
    F: Float + ToPrimitive,
{
    /// Create a new `RCSketch2`
    /// # Arguments
    /// * `input_length` Upper bound on the number of inputs expected.
    /// * `k` Parameter controlling error and buffer size
    pub fn new(k: usize) -> RCSketch2<F> {
        let mut sketch = RCSketch2 {
            buffers: Vec::new(),
            k,
            sketch_size: 0,
            count: 0,
            compaction_counters: Vec::new(),
            number_of_sections: Vec::new(),
            section_sizes: Vec::new(),
        };
        sketch.grow();
        sketch
    }

    /// Calculate the capacity of a buffer
    /// # Arguments
    /// * `h` Index of the buffer
    pub fn calc_buffer_size(&self, h: usize) -> usize {
        return (2 * self.number_of_sections[h] * self.section_sizes[h] as u32) as usize;
    }

    /// Update the number and size of sections of a buffer
    /// # Arguments
    /// `h` Index of the buffer
    pub fn update_sections(&mut self, h: usize) {
        if self.compaction_counters[h] as f32 >= 2.0.powi(self.number_of_sections[h] as i32 - 1) {
            self.number_of_sections[h] *= 2;
            self.section_sizes[h] /= 2.0.sqrt();
        }
    }

    /// Determine the compaction index using an exponential distribution
    /// Should only be called once per compaction as the compaction counter is updated.
    /// # Arguments
    /// * `rc_index` Index of the buffer to determine the compaction index of
    pub fn get_compact_index(&mut self, rc_index: usize) -> usize {
        // Determine the index to remove after
        let num_compaction_sections = self.number_of_sections[rc_index] as usize
            - self.compaction_counters[rc_index].trailing_ones() as usize
            - 1;
        let compact_index = self.buffers[rc_index].len() / 2
            + num_compaction_sections * self.section_sizes[rc_index] as usize;
        self.compaction_counters[rc_index] += 1;
        return compact_index;
    }

    /// Determine the compaction index for a buffer.
    /// Index is always half the target buffer size
    /// # Arguments
    /// * `rc_index` Index of the buffer to determine the compaction index of
    #[inline]
    pub fn get_compact_index_fast(&self, h: usize) -> usize {
        return self.calc_buffer_size(h) / 2;
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
        self.insert_batch(items, fast_compaction, compaction_method);
        self.count += length;
    }

    /// Insert an item into the sketch
    /// # Arguments
    /// * `item` The item to insert
    /// * `fast_compaction` Enables faster compaction in exchange for potentially increased error
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn insert(&mut self, item: F, fast_compaction: bool, compaction_method: CompactionMethod) {
        self.buffers[0].push(item);
        self.sketch_size += 1;
        // If the sketch is full compress
        if self.sketch_size >= self.get_sketch_capacity() {
            self.compress(fast_compaction, compaction_method);
        }
    }

    /// Insert multiple items into the sketch
    /// # Arguments
    /// * `items` The items to insert
    /// * `fast_compaction` Enables faster compaction in exchange for potentially increased error
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn insert_batch(
        &mut self,
        items: &[F],
        fast_compaction: bool,
        compaction_method: CompactionMethod,
    ) {
        // Copy the maximum number of items into the buffer each loop
        let mut current_index = 0;
        while current_index < items.len() {
            assert!(self.get_sketch_capacity() >= self.sketch_size);
            let end = usize::min(
                current_index + (self.get_sketch_capacity() - self.sketch_size),
                items.len(),
            );
            self.buffers[0].extend(&items[current_index..end]);
            self.sketch_size += end - current_index;
            current_index = end;
            // If sketch is full compact and insert into the next buffer
            if self.sketch_size >= self.get_sketch_capacity() {
                self.compress(fast_compaction, compaction_method);
            }
        }
    }

    /// Get the capacity the entire sketch
    /// Sum of the capacity of all the buffers
    pub fn get_sketch_capacity(&self) -> usize {
        return (0..self.buffers.len())
            .map(|i| self.calc_buffer_size(i))
            .sum();
    }

    /// Compact all buffers which exceed the buffer's capacity
    /// * `fast_compaction` Enables faster compaction in exchange for potentially increased error
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn compress(&mut self, fast_compaction: bool, compaction_method: CompactionMethod) {
        for h in 0..self.buffers.len() {
            if self.buffers[h].len() >= self.calc_buffer_size(h) {
                let compaction_index = if fast_compaction {
                    self.get_compact_index_fast(h)
                } else {
                    self.get_compact_index(h)
                };
                let output_items = self.compact_buffer(h, compaction_index, compaction_method);
                self.update_sections(h);
                if self.buffers.len() == h + 1 {
                    self.grow();
                }
                self.sketch_size += output_items.len();
                self.buffers[h + 1].extend(output_items);
            }
        }
    }

    /// Add a new buffer layer
    pub fn grow(&mut self) {
        self.compaction_counters.push(0);
        self.number_of_sections.push(3);
        self.section_sizes.push(self.k as f32);
        self.buffers.push(Vec::with_capacity(
            self.calc_buffer_size(self.buffers.len()),
        ));
    }

    /// Compact a particular buffer and return the output
    /// # Arguments
    /// * `rc_index` Index of the buffer to compact
    /// * `compact_index` Index after which to compact
    /// * `compaction_method` The method with which to compact the removed elements
    pub fn compact_buffer(
        &mut self,
        rc_index: usize,
        compact_index: usize,
        compaction_method: CompactionMethod,
    ) -> Vec<F> {
        // Sort and extract the largest values
        self.buffers[rc_index].sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let upper = self.buffers[rc_index].split_off(compact_index);
        self.sketch_size -= upper.len();

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
    use crate::rc_sketch::rc_sketch2::RCSketch2;
    use crate::traits::Digest;
    use crate::util::gen_asc_vec;
    use crate::util::linear_digest::LinearDigest;
    use approx::assert_relative_eq;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn insert_single_value() {
        let mut sketch = RCSketch2::new(8);
        sketch.add(1.0);
        assert_eq!(sketch.interpolate_rank(1.0), 1);
    }

    #[test]
    fn insert_multiple_values() {
        let mut sketch = RCSketch2::new(8);
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
        let mut sketch = RCSketch2::new(8);
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
        let mut sketch = RCSketch2::new(8);
        sketch.add_buffer(&vec![1.0]);
        assert_eq!(sketch.interpolate_rank(1.0), 1);
    }

    #[test]
    fn insert_batch_multiple_values() {
        let mut sketch = RCSketch2::new(8);
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
        let mut sketch = RCSketch2::new(8);
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
        let mut digest = RCSketch2::new(200);
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
        let mut digest = RCSketch2::new(200);
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
        let mut digest = RCSketch2::new(320);
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
        let mut digest = RCSketch2::new(315);
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
        let mut sketch = RCSketch2::new(16);
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
