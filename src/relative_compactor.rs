use rand::thread_rng;
use rand::Rng;

#[derive(Debug)]
pub struct RCSketch {
    pub buffers: Vec<Vec<f64>>,
    pub count: u64,
    pub sorted: bool,
}

impl RCSketch {
    pub fn new() -> RCSketch {
        RCSketch {
            buffers: Vec::new(),
            count: 0,
            sorted: false,
        }
    }

    pub fn insert(&mut self, item: f64, rc_index: usize) {
        self.sorted = false;
        if self.buffers.len() <= rc_index {
            self.buffers.push(Vec::new());
        }
        self.buffers[rc_index].push(item);
        if self.buffers[rc_index].len() >= 200 {
            for item in self.compact(rc_index) {
                self.insert(item, rc_index + 1);
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

    pub fn sort(&mut self) {
        for buffer in &mut self.buffers {
            buffer.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        }
        self.sorted = true;
    }

    pub fn interpolate_rank(&mut self, rank_item: f64) -> u64 {
        if self.sorted {
            self.sort()
        }
        let mut rank = 0;
        for i in 0..self.buffers.len() {
            rank += self.buffers[i]
                .iter()
                .filter(|x| **x <= rank_item)
                .collect::<Vec<&f64>>()
                .len() as u64
                * (2 as u64).pow(i as u32);
        }
        rank
    }
}

#[cfg(test)]
mod test {
    use crate::relative_compactor::RCSketch;
    use approx::assert_relative_eq;

    #[test]
    fn insert_single_value() {
        let mut sketch = RCSketch::new();
        sketch.insert(1.0, 0);
    }

    #[test]
    fn insert_multiple_values() {
        let mut sketch = RCSketch::new();
        (0..1000).map(|x| sketch.insert(x as f64, 0)).for_each(drop);

        println!("{:?}", sketch);
        assert_eq!(sketch.interpolate_rank(0.0), 1);
        assert_eq!(sketch.interpolate_rank(1.0), 2);
        assert_eq!(sketch.interpolate_rank(10.0), 11);
        assert_relative_eq!(
            sketch.interpolate_rank(1000.0) as f64,
            1000 as f64,
            epsilon = 30.0
        );
    }

    #[test]
    fn insert_descending_multiple_values() {
        let mut sketch = RCSketch::new();
        (0..1000)
            .map(|x| sketch.insert(999.0 - x as f64, 0))
            .for_each(drop);

        println!("{:?}", sketch);
        sketch.sort();
        println!("{:?}", sketch);
        assert_eq!(sketch.interpolate_rank(0.0), 1);
        assert_eq!(sketch.interpolate_rank(1.0), 2);
        assert_relative_eq!(
            sketch.interpolate_rank(1000.0) as f64,
            1000 as f64,
            epsilon = 30.0
        );
    }
}
