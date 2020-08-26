use rand::thread_rng;
use rand::Rng;

#[derive(Debug)]
pub struct RCSketch {
    pub buffers: Vec<Vec<f64>>,
    pub buffer_size: usize,
    pub count: u64,
}

impl RCSketch {
    pub fn new(buffer_size: usize) -> RCSketch {
        RCSketch {
            buffers: Vec::new(),
            buffer_size,
            count: 0,
        }
    }

    pub fn insert(&mut self, item: f64) {
        self.insert_at_rc(item, 0);
        self.count += 1;
    }

    pub fn insert_at_rc(&mut self, item: f64, rc_index: usize) {
        if self.buffers.len() <= rc_index {
            self.buffers.push(Vec::with_capacity(self.buffer_size));
        }
        self.buffers[rc_index].push(item);
        if self.buffers[rc_index].len() >= self.buffer_size {
            for item in self.compact(rc_index) {
                self.insert_at_rc(item, rc_index + 1);
            }
        }
    }

    pub fn compact(&mut self, rc_index: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        let compact_index = rng.gen_range(
            self.buffers[rc_index].len() / 2,
            self.buffers[rc_index].len(),
        );
        self.buffers[rc_index].sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let pivot = self.buffers[rc_index][compact_index];

        let (lower, upper): (Vec<f64>, Vec<f64>) =
            self.buffers[rc_index].iter().partition(|x| **x < pivot);
        self.buffers[rc_index] = lower;
        upper
            .iter()
            .enumerate()
            .filter(|(pos, _value)| pos % 2 == 0)
            .map(|(_pos, value)| *value)
            .collect()
    }

    pub fn interpolate_rank(&mut self, rank_item: f64) -> u64 {
        let mut rank = 0;
        for i in 0..self.buffers.len() {
            rank += self.buffers[i]
                .iter()
                .filter(|x| **x <= rank_item)
                .collect::<Vec<&f64>>()
                .len() as u64
                * (1 << i);
        }
        rank
    }

    pub fn interpolate(&mut self, rank_item: f64) -> f64 {
        self.interpolate_rank(rank_item) as f64 / self.count as f64
    }
}

#[cfg(test)]
mod test {
    use crate::relative_compactor::RCSketch;
    use approx::assert_relative_eq;

    #[test]
    fn insert_single_value() {
        let mut sketch = RCSketch::new(64);
        sketch.insert(1.0);
        assert_eq!(sketch.interpolate_rank(1.0), 1);
    }

    #[test]
    fn insert_multiple_values() {
        let mut sketch = RCSketch::new(256);
        (0..1000).map(|x| sketch.insert(x as f64)).for_each(drop);

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
        let mut sketch = RCSketch::new(256);
        (0..1000)
            .map(|x| sketch.insert(999.0 - x as f64))
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
}
