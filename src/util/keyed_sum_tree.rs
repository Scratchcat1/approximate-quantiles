// use std::cell::RefCell;
// use std::rc::Rc;
use std::cmp::Ordering::{Equal, Greater, Less};

struct KeyedSumNode {
    key: f64,
    weight: f64,
    sum: f64,
    left_child: Option<Box<KeyedSumNode>>,
    right_child: Option<Box<KeyedSumNode>>,
}

impl KeyedSumNode {
    pub fn insert(&mut self, insert_key: f64, weight: f64) {
        if insert_key < self.key {
            match &mut self.left_child {
                None => {
                    self.left_child = Some(Box::new(KeyedSumNode {
                        key: insert_key,
                        weight,
                        sum: weight,
                        left_child: None,
                        right_child: None,
                    }))
                }
                Some(child) => child.insert(insert_key, weight),
            }
        } else if self.key < insert_key {
            match &mut self.right_child {
                None => {
                    self.right_child = Some(Box::new(KeyedSumNode {
                        key: insert_key,
                        weight,
                        sum: weight,
                        left_child: None,
                        right_child: None,
                    }))
                }
                Some(child) => child.insert(insert_key, weight),
            }
        } else {
            panic!("KeyedSumNode is not designed to have identical key nodes");
        }
        self.sum += weight;
    }

    // pub fn delete(&mut self, target_key: f64) -> Result<f64, ()> {
    //     if target_key < self.key {
    //         return match &mut self.left_child {
    //             None => Err(()),
    //             Some(child) => {
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

    pub fn remove_leftmost_node(&mut self) {}

    pub fn take_left_child(&mut self) -> Option<Box<KeyedSumNode>> {
        return self.left_child.take();
    }

    pub fn take_right_child(&mut self) -> Option<Box<KeyedSumNode>> {
        return self.right_child.take();
    }
}

struct KeyedSumTree {
    root: Option<Box<KeyedSumNode>>,
}

// Based on https://codereview.stackexchange.com/questions/133209/binary-tree-implementation-in-rust
// https://stackoverflow.com/questions/64043682/how-to-write-a-delete-function-for-a-binary-tree-in-rust
impl KeyedSumTree {
    pub fn find(&self, target_key: f64) -> &Option<Box<KeyedSumNode>> {
        let mut current = &self.root;
        while let Some(ref node) = *current {
            match node.key.partial_cmp(&target_key).unwrap() {
                Less => current = &node.left_child,
                Greater => current = &node.left_child,
                Equal => return current,
            }
        }
        current
    }

    pub fn find_mut(&mut self, target_key: f64) -> &mut Option<Box<KeyedSumNode>> {
        let mut anchor = &mut self.root;
        loop {
            match { anchor } {
                &mut Some(ref mut node) if target_key != node.key => {
                    anchor = if target_key < node.key {
                        &mut node.left_child
                    } else {
                        &mut node.right_child
                    }
                }

                other @ &mut Some(_) | other @ &mut None => return other,
            }
        }
    }

    pub fn delete(&mut self, target_key: f64) {
        if let Some(target_node) = self.find(target_key) {
            let weight = target_node.weight;

            // Remove the target node weight from all of the parent nodes
            let mut current = &mut self.root;
            while let Some(ref mut node) = *current {
                node.weight -= weight;
                match node.key.partial_cmp(&target_key).unwrap() {
                    Less => current = &mut node.left_child,
                    Greater => current = &mut node.left_child,
                    Equal => break,
                }
            }
            // if let Some(successor(mut next: &mut Option<Box<KeyedSumNode>>))
        }
    }
}

// fn successor(mut next: &mut Option<Box<KeyedSumNode>>) -> &mut Option<Box<KeyedSumNode>> {
//     loop {
//         match { next } {
//             &mut Some(ref mut node) if node.left_child.is_some() => next = &mut node.left_child,
//             other @ &mut Some(_) => return other,
//             _ => unreachable!(),
//         }
//     }
// }

fn successor(mut next: &mut Option<Box<KeyedSumNode>>) -> &mut Option<Box<KeyedSumNode>> {
    loop {
        match { next } {
            &mut Some(ref mut node) if node.left_child.is_some() => next = &mut node.left_child,
            other @ &mut Some(_) => return other,
            _ => unreachable!(),
        }
    }
}

// struct KeyedSumNode {
//     key: f64,
//     left_sum: f64,
// }

// struct KeyedSumTree {
//     // An unbalanced tree would be very space inefficient
//     tree: Vec<Option<KeyedSumNode>>,
// }

// impl KeyedSumTree {
//     pub fn insert(&mut self, key: f64, value: f64) {}
// }
