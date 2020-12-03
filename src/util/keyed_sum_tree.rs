// use std::cell::RefCell;
// use std::rc::Rc;
use crate::t_digest::centroid::Centroid;
use std::cmp::Ordering::{Equal, Greater, Less};

#[derive(Debug)]
struct KeyedSumNode {
    key: f64,
    weight: f64,
    sum: f64,
    left_child: Option<Box<KeyedSumNode>>,
    right_child: Option<Box<KeyedSumNode>>,
}

impl KeyedSumNode {
    pub fn new(key: f64, weight: f64) -> Self {
        KeyedSumNode {
            key,
            weight,
            sum: weight,
            left_child: None,
            right_child: None,
        }
    }

    pub fn less_than_sum(&self, target_key: f64) -> f64 {
        match (
            target_key.partial_cmp(&self.key).unwrap(),
            &self.left_child,
            &self.right_child,
        ) {
            (Less, None, _) => 0.0,
            (Less, Some(left), _) => left.less_than_sum(target_key),
            (Equal, None, _) => 0.0,
            (Equal, Some(left), _) => left.less_than_sum(target_key),
            (Greater, None, None) => self.weight,
            (Greater, Some(left), None) => self.weight + left.sum,
            (Greater, None, Some(right)) => self.weight + right.less_than_sum(target_key),
            (Greater, Some(left), Some(right)) => {
                self.weight + left.sum + right.less_than_sum(target_key)
            }
        }
    }

    pub fn insert(&mut self, insert_key: f64, weight: f64) {
        if insert_key < self.key {
            match &mut self.left_child {
                None => self.left_child = Some(Box::new(KeyedSumNode::new(insert_key, weight))),
                Some(child) => child.insert(insert_key, weight),
            }
        } else if self.key < insert_key {
            match &mut self.right_child {
                None => self.right_child = Some(Box::new(KeyedSumNode::new(insert_key, weight))),
                Some(child) => child.insert(insert_key, weight),
            }
        } else {
            panic!("KeyedSumNode is not designed to have identical key nodes");
        }
        self.sum += weight;
    }

    pub fn update(&mut self, target_key: f64, new_weight: f64) -> Option<f64> {
        return if target_key < self.key {
            match &mut self.left_child {
                None => None,
                Some(child) => {
                    let old_weight_option = child.update(target_key, new_weight);
                    if let Some(old_weight) = old_weight_option {
                        self.sum -= old_weight;
                        self.sum += new_weight;
                    }
                    old_weight_option
                }
            }
        } else if self.key < target_key {
            match &mut self.right_child {
                None => None,
                Some(child) => {
                    let old_weight_option = child.update(target_key, new_weight);
                    if let Some(old_weight) = old_weight_option {
                        self.sum -= old_weight;
                        self.sum += new_weight;
                    }
                    old_weight_option
                }
            }
        } else {
            let old_weight_option = Some(self.weight);
            self.sum -= self.weight;
            self.sum += new_weight;
            self.weight = new_weight;
            old_weight_option
        };
    }

    pub fn delete(mut self: Box<Self>, target_key: f64) -> Option<Box<KeyedSumNode>> {
        match target_key.partial_cmp(&self.key).unwrap() {
            Less => {
                if let Some(left) = self.left_child.take() {
                    self.left_child = Self::delete(left, target_key);
                }
                return Some(self);
            }
            Greater => {
                if let Some(right) = self.right_child.take() {
                    self.right_child = Self::delete(right, target_key);
                }
                return Some(self);
            }
            Equal => match (self.left_child.take(), self.right_child.take()) {
                (None, None) => None,
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (Some(mut left), Some(right)) => {
                    if let Some(mut rightmost) = left.rightmost_child() {
                        rightmost.left_child = Some(left);
                        rightmost.right_child = Some(right);
                        Some(rightmost)
                    } else {
                        left.right_child = Some(right);
                        Some(left)
                    }
                }
            },
        }
    }

    //  Returns the rightmost child, unless the node itself is that child.
    fn rightmost_child(&mut self) -> Option<Box<KeyedSumNode>> {
        match self.right_child {
            Some(ref mut right) => {
                if let Some(t) = right.rightmost_child() {
                    Some(t)
                } else {
                    let mut r = self.right_child.take();
                    if let Some(ref mut r) = r {
                        self.right_child = std::mem::replace(&mut r.left_child, None);
                    }
                    r
                }
            }
            None => None,
        }
    }

    fn sorted_vec_key(&self, vec: &mut Vec<f64>) {
        if let Some(child) = &self.left_child {
            child.sorted_vec_key(vec);
        }
        vec.push(self.weight);
        if let Some(child) = &self.right_child {
            child.sorted_vec_key(vec);
        }
    }

    fn closest_keys(&self, target_key: f64, vec: &mut Vec<f64>) {
        vec.push(self.weight);
        match target_key.partial_cmp(&self.key).unwrap() {
            Less => {
                if let Some(child) = &self.right_child {
                    child.closest_keys(target_key, vec);
                }
            }
            Equal => {}
            Greater => {
                if let Some(child) = &self.left_child {
                    child.closest_keys(target_key, vec);
                }
            }
        }
    }

    // pub fn delete(&mut self, target_key: f64) -> Result<f64, ()> {
    //     if target_key < self.key {
    //         return match &mut self.left_child {
    //             None => Err(()),
    //             Some(child) => {self
    //                 let weight = child.delete(target_key)?;
    //                 self.sum -= weight;
    //                 Ok(weight)
    //             }
    //         };
    //     } else if self.key < target_key {
    //         return match &mut self.right_child {
    //             None => Err(()),
    //             Some(child) => {
    //                 let weight = child.delete(target_key)?;
    //                 self.sum -= weight;
    //                 Ok(weight)
    //             }
    //         };
    //     } else {
    //         // Equal
    //         if self.left_child.is_none() && self.right_child.is_none()
    //         return Err(());
    //     }
    // }

    // pub fn remove_leftmost_node(&mut self) {}

    // pub fn take_left_child(&mut self) -> Option<Box<KeyedSumNode>> {
    //     return self.left_child.take();
    // }

    // pub fn take_right_child(&mut self) -> Option<Box<KeyedSumNode>> {
    //     return self.right_child.take();
    // }
}

#[derive(Debug)]
pub struct KeyedSumTree {
    root: Option<Box<KeyedSumNode>>,
}

// Based on https://codereview.stackexchange.com/questions/133209/binary-tree-implementation-in-rust
// https://stackoverflow.com/questions/64043682/how-to-write-a-delete-function-for-a-binary-tree-in-rust
impl KeyedSumTree {
    // pub fn find(&self, target_key: f64) -> &Option<Box<KeyedSumNode>> {
    //     let mut current = &self.root;
    //     while let Some(ref node) = *current {
    //         match node.key.partial_cmp(&target_key).unwrap() {
    //             Less => current = &node.left_child,
    //             Greater => current = &node.left_child,
    //             Equal => return current,
    //         }
    //     }
    //     current
    // }

    pub fn new() -> Self {
        KeyedSumTree { root: None }
    }

    pub fn less_than_sum(&self, key: f64) -> Option<f64> {
        if let Some(root) = &self.root {
            // Some(root.less_than_sum(key))
            let mut current = root;
            let mut sum = 0.0;
            loop {
                match (
                    key.partial_cmp(&current.key).unwrap(),
                    &current.left_child,
                    &current.right_child,
                ) {
                    (Less, None, _) => break,
                    (Less, Some(left), _) => current = left,
                    (Equal, None, _) => break,
                    (Equal, Some(left), _) => current = left,
                    (Greater, None, None) => {
                        sum += current.weight;
                        break;
                    }
                    (Greater, Some(left), None) => {
                        sum += current.weight + left.sum;
                        break;
                    }
                    (Greater, None, Some(right)) => {
                        sum += current.weight;
                        current = right;
                    }
                    (Greater, Some(left), Some(right)) => {
                        sum += current.weight + left.sum;
                        current = right;
                    }
                };
            }
            Some(sum)
        } else {
            None
        }
    }

    // pub fn insert(&mut self, key: f64, weight: f64) {
    //     if let Some(root) = &mut self.root {
    //         root.insert(key, weight);
    //     } else {
    //         self.root = Some(Box::new(KeyedSumNode::new(key, weight)));
    //     }
    // }

    pub fn insert(&mut self, key: f64, weight: f64) {
        if let Some(root) = &mut self.root {
            // root.insert(key, weight);
            let mut current = root;
            loop {
                current.sum += weight;
                if key < current.key {
                    match current.left_child.is_none() {
                        true => {
                            current.left_child = Some(Box::new(KeyedSumNode::new(key, weight)));
                            break;
                        }
                        false => {
                            current = current.left_child.as_mut().unwrap();
                            // current.weight += weight;
                        }
                    }
                } else if current.key < key {
                    match current.right_child.is_none() {
                        true => {
                            current.right_child = Some(Box::new(KeyedSumNode::new(key, weight)));
                            break;
                        }
                        false => {
                            current = current.right_child.as_mut().unwrap();
                            // current.weight += weight;
                        }
                    }
                } else {
                    panic!("KeyedSumNode is not designed to have identical key nodes");
                }
            }
        } else {
            self.root = Some(Box::new(KeyedSumNode::new(key, weight)));
        }
    }

    pub fn update(&mut self, key: f64, new_weight: f64) -> Option<f64> {
        if let Some(root) = &mut self.root {
            root.update(key, new_weight)
        } else {
            None
        }
    }

    pub fn delete(&mut self, target_key: f64) {
        if let Some(root) = self.root.take() {
            self.root = KeyedSumNode::delete(root, target_key);
        }
    }

    pub fn sorted_vec_key(&self) -> Vec<f64> {
        if let Some(root) = &self.root {
            let mut vec = Vec::new();
            root.sorted_vec_key(&mut vec);
            vec
        } else {
            Vec::new()
        }
    }

    pub fn closest_keys(&self, target_key: f64) -> Vec<f64> {
        if let Some(root) = &self.root {
            let mut vec = Vec::new();
            root.closest_keys(target_key, &mut vec);
            let min = *vec
                .iter()
                .min_by(|a, b| {
                    (**a - target_key)
                        .abs()
                        .partial_cmp(&(**b - target_key).abs())
                        .unwrap()
                })
                .unwrap();
            vec.into_iter()
                .filter(|x| (x - target_key).abs() == min)
                .collect()
        } else {
            Vec::new()
        }
    }
}

impl From<&[Centroid]> for KeyedSumTree {
    fn from(slice: &[Centroid]) -> Self {
        let mut tree = KeyedSumTree::new();
        for centroid in slice {
            tree.insert(centroid.mean, centroid.weight);
        }
        return tree;
    }
}

#[cfg(test)]
mod test {
    use crate::util;
    use crate::util::keyed_sum_tree::KeyedSumTree;
    use approx::assert_relative_eq;

    #[test]
    fn manual() {
        let mut tree = KeyedSumTree::new();
        assert_eq!(tree.less_than_sum(0.0), None);
        assert_eq!(tree.less_than_sum(1_000_000.0), None);

        tree.insert(1.0, 1.0);
        assert_relative_eq!(tree.less_than_sum(1.0).unwrap(), 0.0);
        assert_relative_eq!(tree.less_than_sum(2.0).unwrap(), 1.0);

        tree.delete(1.0);
        assert_eq!(tree.less_than_sum(0.0), None);
        assert_eq!(tree.less_than_sum(1_000_000.0), None);

        tree.insert(1.0, 1.0);
        tree.insert(13.0, 100.0);
        tree.insert(25.0, 1.0);
        tree.insert(-100.0, 5.0);
        println!("{:?}", tree);

        assert_relative_eq!(tree.less_than_sum(-101.0).unwrap(), 0.0);
        assert_relative_eq!(tree.less_than_sum(-100.0).unwrap(), 0.0);
        assert_relative_eq!(tree.less_than_sum(-99.0).unwrap(), 5.0);

        assert_relative_eq!(tree.less_than_sum(0.0).unwrap(), 5.0);
        assert_relative_eq!(tree.less_than_sum(1.0).unwrap(), 5.0);
        assert_relative_eq!(tree.less_than_sum(2.0).unwrap(), 6.0);

        assert_relative_eq!(tree.less_than_sum(12.0).unwrap(), 6.0);
        assert_relative_eq!(tree.less_than_sum(13.0).unwrap(), 6.0);
        assert_relative_eq!(tree.less_than_sum(14.0).unwrap(), 106.0);

        assert_relative_eq!(tree.less_than_sum(24.0).unwrap(), 106.0);
        assert_relative_eq!(tree.less_than_sum(25.0).unwrap(), 106.0);
        assert_relative_eq!(tree.less_than_sum(26.0).unwrap(), 107.0);
    }

    #[test]
    fn uniform() {
        let mut tree = KeyedSumTree::new();
        assert_eq!(tree.less_than_sum(0.0), None);
        assert_eq!(tree.less_than_sum(1_000_000.0), None);

        let mut centroids = util::gen_uniform_centroid_random_weight_vec(1_000);

        for centroid in centroids.iter() {
            tree.insert(centroid.mean, centroid.weight);
        }

        centroids.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());

        let mut sum = 0.0;
        for centroid in centroids.iter() {
            assert_relative_eq!(
                tree.less_than_sum(centroid.mean).unwrap(),
                sum,
                epsilon = 1e-7
            );
            sum += centroid.weight;
        }

        // Delete half of the centroids
        let removed = centroids.split_off(centroids.len() / 2);

        for centroid in removed {
            tree.delete(centroid.mean);
        }

        let mut sum = 0.0;
        for centroid in centroids.iter() {
            assert_relative_eq!(
                tree.less_than_sum(centroid.mean).unwrap(),
                sum,
                epsilon = 1e-7
            );
            sum += centroid.weight;
        }
    }
}
