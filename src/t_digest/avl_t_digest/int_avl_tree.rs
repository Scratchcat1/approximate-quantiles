use crate::t_digest::avl_t_digest::int_avl_tree_store::IntAVLTreeStore;
use crate::t_digest::avl_t_digest::node_allocator::NodeAllocator;
use crate::t_digest::avl_t_digest::{node_id_to_option, NIL};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct IntAVLTree<S, T>
where
    S: IntAVLTreeStore<T>,
{
    node_allocator: NodeAllocator,
    root: u32,
    parent: Vec<u32>,
    left: Vec<u32>,
    right: Vec<u32>,
    depth: Vec<u8>,
    store: S,
    phantom: std::marker::PhantomData<T>,
}

impl<S, T> Default for IntAVLTree<S, T>
where
    S: IntAVLTreeStore<T>,
    T: Copy,
{
    fn default() -> Self {
        Self::new(16)
    }
}

impl<S, T> IntAVLTree<S, T>
where
    S: IntAVLTreeStore<T>,
    T: Copy,
{
    pub fn new(capacity: usize) -> Self {
        let mut tree = IntAVLTree {
            node_allocator: NodeAllocator::default(),
            root: NIL,
            parent: Vec::with_capacity(capacity),
            left: Vec::with_capacity(capacity),
            right: Vec::with_capacity(capacity),
            depth: Vec::with_capacity(capacity),
            store: S::with_capacity(capacity as u32),
            phantom: std::marker::PhantomData,
        };
        tree.resize(capacity);
        tree
    }

    /// Resize all arrays to the `new_capacity`
    /// The extra node slots are filled with NIL
    /// The extra data slots are filled with the default value of T
    fn resize(&mut self, new_capacity: usize) {
        self.parent.resize(new_capacity, NIL);
        self.left.resize(new_capacity, NIL);
        self.right.resize(new_capacity, NIL);
        self.depth.resize(new_capacity, 0);
        self.store.resize(new_capacity as u32);
    }

    /// Returns the capacity
    fn capacity(&self) -> usize {
        self.parent.len()
    }

    /// Return the current root of the tree
    pub fn get_root(&self) -> u32 {
        self.root
    }

    /// Return the parent of the node
    #[inline]
    pub fn get_parent(&self, node: u32) -> u32 {
        self.parent[node as usize]
    }

    /// Set the parent of the node
    #[inline]
    pub fn set_parent(&mut self, node: u32, item: u32) {
        assert!(node != NIL);
        self.parent[node as usize] = item;
    }

    /// Return the left child of the node
    #[inline]
    pub fn get_left(&self, node: u32) -> u32 {
        self.left[node as usize]
    }

    /// Set the left child of the node
    #[inline]
    pub fn set_left(&mut self, node: u32, item: u32) {
        assert!(node != NIL);
        self.left[node as usize] = item;
    }

    /// Return the right child of the node
    #[inline]
    pub fn get_right(&self, node: u32) -> u32 {
        self.right[node as usize]
    }

    /// Set the right child of the node
    #[inline]
    pub fn set_right(&mut self, node: u32, item: u32) {
        assert!(node != NIL);
        self.right[node as usize] = item;
    }

    /// Return the depth of the node
    #[inline]
    pub fn get_depth(&self, node: u32) -> u8 {
        self.depth[node as usize]
    }

    /// Set the depth of the node
    #[inline]
    pub fn set_depth(&mut self, node: u32, item: u8) {
        assert!(node != NIL);
        self.depth[node as usize] = item;
    }

    /// Return a reference to the store
    pub fn get_store(&self) -> &S {
        &self.store
    }

    /// Return a mutable reference to the store
    pub fn get_mut_store(&mut self) -> &mut S {
        &mut self.store
    }

    /// Return the size of the tree
    pub fn size(&self) -> u32 {
        self.node_allocator.size()
    }

    /// Returns the least node under `node` or None if not found
    pub fn first(&self, mut node: u32) -> Option<u32> {
        if node == NIL {
            return None;
        }

        loop {
            let left = self.get_left(node);
            if left == NIL {
                break;
            }
            node = left;
        }
        return if node == NIL { None } else { Some(node) };
    }

    /// Returns the greatest node under `node` or None if not found
    pub fn last(&self, mut node: u32) -> Option<u32> {
        if node == NIL {
            return None;
        }

        loop {
            let right = self.get_right(node);
            if right == NIL {
                break;
            }
            node = right;
        }

        return if node == NIL { None } else { Some(node) };
    }

    /// Returns the least node that is strictly greater than `node` or None if not found
    pub fn next(&self, mut node: u32) -> Option<u32> {
        let right = self.get_right(node);
        if right != NIL {
            return self.first(right);
        } else {
            let mut parent = self.get_parent(node);
            while parent != NIL && node == self.get_right(parent) {
                node = parent;
                parent = self.get_parent(parent);
            }
            return if parent == NIL { None } else { Some(parent) };
        }
    }

    /// Returns the greatest node that is strictly less than `node` or None if not found
    pub fn prev(&self, mut node: u32) -> Option<u32> {
        let left = self.get_left(node);
        if left != NIL {
            return self.last(left);
        } else {
            let mut parent = self.get_parent(node);
            while parent != NIL && node == self.get_left(parent) {
                node = parent;
                parent = self.get_parent(parent);
            }
            return if parent == NIL { None } else { Some(parent) };
        }
    }

    /// Add an item to the tree
    pub fn add_by(&mut self, item: T) {
        if self.root == NIL {
            self.root = self.node_allocator.new_node();
            self.store.copy(self.root, item);
            self.fix_aggregates(self.root);
        } else {
            let mut node = self.root;
            assert!(self.get_parent(self.root) == NIL);
            let mut comparison;
            let mut parent;
            loop {
                comparison = self.store.compare(node, item);
                match comparison {
                    Ordering::Less => {
                        parent = node;
                        node = self.get_left(node);
                    }
                    Ordering::Greater => {
                        parent = node;
                        node = self.get_right(node);
                    }
                    _ => {
                        todo!();
                    }
                }
                if node == NIL {
                    break;
                }
            }

            let node = self.node_allocator.new_node();
            if node as usize >= self.capacity() {
                self.resize(2 * node as usize);
            }

            self.store.copy(node, item);
            self.set_parent(node, parent);

            match comparison {
                Ordering::Less => {
                    self.set_left(parent, node);
                }
                Ordering::Greater => {
                    self.set_right(parent, node);
                }
                Ordering::Equal => {
                    panic!("Comparison resulted in equal when only less or greater should be possible. Please file a bug report.");
                }
            }

            self.rebalance(node);
        }
    }

    /// Find a node in the tree
    /// Returns None if not found
    pub fn find_by(&self, item: T) -> Option<T> {
        let mut node = self.root;
        while node != NIL {
            match self.store.compare(node, item) {
                Ordering::Less => {
                    node = self.get_left(node);
                }
                Ordering::Greater => {
                    node = self.get_right(node);
                }
                Ordering::Equal => {
                    return Some(self.store.read(node));
                }
            }
        }
        return None;
    }

    /// Update a node with new data
    pub fn update_by(&mut self, node: u32, item: T) {
        let prev = self.prev(node);
        let next = self.next(node);
        if (prev.is_none() || self.store.compare(prev.unwrap(), item) == Ordering::Greater)
            && (next.is_none() || self.store.compare(next.unwrap(), item) == Ordering::Less)
        {
            self.store.copy(node, item);
            let mut n = node;
            while n != NIL {
                self.fix_aggregates(n);
                n = self.get_parent(n);
            }
        } else {
            self.remove(node);
            self.add_by(item);
        }
    }

    /// Remove the specified node from the tree
    /// # Panics
    /// Panics if `node` is not in the tree
    pub fn remove(&mut self, node: u32) {
        if node == NIL {
            panic!("Cannot remove a node with is not in the tree: NIL node");
        }

        if self.get_left(node) != NIL && self.get_right(node) != NIL {
            // inner node, two children
            let next = self.next(node);
            assert!(next.is_some());
            self.swap(node, next.unwrap());
        }

        assert!(self.get_left(node) == NIL || self.get_right(node) == NIL);

        let parent = self.get_parent(node);
        let mut child = self.get_left(node);
        if child == NIL {
            child = self.get_right(node);
        }

        if child == NIL {
            // no children
            if node == self.root {
                assert!(self.size() == 1);
                self.root = NIL;
            } else {
                if node == self.get_left(parent) {
                    self.set_left(parent, NIL);
                } else {
                    assert!(node == self.get_right(parent));
                    self.set_right(parent, NIL)
                }
            }
        } else {
            // one child
            if node == self.root {
                assert!(self.size() == 2);
                self.root = child;
            } else if node == self.get_left(parent) {
                self.set_left(parent, child);
            } else {
                assert!(node == self.get_right(parent));
                self.set_right(parent, child);
            }
            self.set_parent(child, parent);
        }

        self.release(node);
        self.rebalance(parent);
    }

    /// Release the node
    /// Marks the node as unused in the node allocator
    fn release(&mut self, node: u32) {
        self.set_left(node, NIL);
        self.set_right(node, NIL);
        self.set_parent(node, NIL);
        self.node_allocator.release(node);
    }

    /// Swap two nodes
    fn swap(&mut self, node1: u32, node2: u32) {
        assert!(node1 != NIL && node2 != NIL);

        let parent1 = self.get_parent(node1);
        let parent2 = self.get_parent(node2);

        if parent1 != NIL {
            if node1 == self.get_left(parent1) {
                self.set_left(parent1, node2);
            } else {
                assert!(node1 == self.get_right(parent1));
                self.set_right(parent1, node2);
            }
        } else {
            assert!(self.root == node1);
            self.root = node2;
        }

        if parent2 != NIL {
            if node2 == self.get_left(parent2) {
                self.set_left(parent2, node1);
            } else {
                assert!(node2 == self.get_right(parent2));
                self.set_right(parent2, node1);
            }
        } else {
            assert!(self.root == node2);
            self.root = node1;
        }

        self.set_parent(node1, parent2);
        self.set_parent(node2, parent1);

        let left1 = self.get_left(node1);
        let left2 = self.get_left(node2);
        self.set_left(node1, left2);
        if left2 != NIL {
            self.set_parent(left2, node1);
        }
        self.set_left(node2, left1);
        if left1 != NIL {
            self.set_parent(left1, node2);
        }

        let right1 = self.get_right(node1);
        let right2 = self.get_right(node2);
        self.set_right(node1, right2);
        if right2 != NIL {
            self.set_parent(right2, node1);
        }
        self.set_right(node2, right1);
        if right1 != NIL {
            self.set_parent(right1, node2);
        }

        let depth1 = self.get_depth(node1);
        let depth2 = self.get_depth(node2);
        self.set_depth(node1, depth2);
        self.set_depth(node2, depth1);
    }

    /// Returns the balance of the avl tree node
    fn balance_factor(&self, node: u32) -> i16 {
        return self.get_depth(self.get_left(node)) as i16
            - self.get_depth(self.get_right(node)) as i16;
    }

    /// Rebalance the node
    fn rebalance(&mut self, node: u32) {
        let mut n = node;
        while n != NIL {
            let p = self.get_parent(n);

            self.fix_aggregates(n);

            match self.balance_factor(n) {
                -2 => {
                    let right = self.get_right(n);
                    if self.balance_factor(right) == 1 {
                        self.rotate_right(right);
                    }
                    self.rotate_left(n);
                }
                2 => {
                    let left = self.get_left(n);
                    if self.balance_factor(left) == -1 {
                        self.rotate_left(left);
                    }
                    self.rotate_right(n);
                }
                -1 | 0 | 1 => { // Balance is alright
                }
                _ => {
                    panic!("AVL Tree has a balance < -2 or > 2. This should not be possible. Please file a bug report");
                }
            }

            n = p;
        }
    }

    /// Fix the depth for a node
    fn fix_depth(&mut self, node: u32) {
        let left_depth = self.get_depth(self.get_left(node));
        let right_depth = self.get_depth(self.get_right(node));
        self.set_depth(node, 1 + u8::max(left_depth, right_depth));
    }

    /// Rotate the subtree under node `n` left
    fn rotate_left(&mut self, n: u32) {
        let r = self.get_right(n);
        let lr = self.get_left(r);

        self.set_right(n, lr);
        if lr != NIL {
            self.set_parent(lr, n);
        }

        let p = self.get_parent(n);
        self.set_parent(r, p);
        if p == NIL {
            self.root = r;
        } else if self.get_left(p) == n {
            self.set_left(p, r);
        } else {
            assert!(self.get_right(p) == n);
            self.set_right(p, r);
        }

        self.set_left(r, n);
        self.set_parent(n, r);
        self.fix_aggregates(n);
        self.fix_aggregates(self.get_parent(n));
    }

    /// Rotate the subtree under node `n` right
    fn rotate_right(&mut self, n: u32) {
        let l = self.get_left(n);
        let rl = self.get_right(l);

        self.set_left(n, rl);
        if rl != NIL {
            self.set_parent(rl, n);
        }

        let p = self.get_parent(n);
        self.set_parent(l, p);
        if p == NIL {
            self.root = l;
        } else if self.get_right(p) == n {
            self.set_right(p, l);
        } else {
            assert!(self.get_left(p) == n);
            self.set_left(p, l);
        }

        self.set_right(l, n);
        self.set_parent(n, l);
        self.fix_aggregates(n);
        self.fix_aggregates(self.get_parent(n));
    }

    pub fn fix_aggregates(&mut self, node: u32) {
        self.fix_depth(node);
        let left = self.get_left(node);
        let right = self.get_right(node);

        let left_child = if left != NIL { Some(left) } else { None };
        let right_child = if right != NIL { Some(right) } else { None };
        self.store.fix_aggregates(node, left_child, right_child);
    }

    /// Returns the last node id which is less than `item`
    pub fn floor(&self, item: T) -> Option<u32> {
        let mut floor = NIL;
        let mut node = self.root;
        while node != NIL {
            match self.store.compare(node, item) {
                Ordering::Less | Ordering::Equal => {
                    node = self.get_left(node);
                }
                Ordering::Greater => {
                    floor = node;
                    node = self.get_right(node);
                }
            }
        }
        node_id_to_option(floor)
    }
}

impl<'a, S, T> IntoIterator for &'a IntAVLTree<S, T>
where
    S: IntAVLTreeStore<T>,
    T: Copy,
{
    type Item = u32;
    type IntoIter = IntAVLTreeRefIterator<'a, S, T>;

    fn into_iter(self) -> IntAVLTreeRefIterator<'a, S, T> {
        IntAVLTreeRefIterator::new(&self)
    }
}

pub struct IntAVLTreeRefIterator<'a, S, T>
where
    S: IntAVLTreeStore<T>,
{
    current_node: Option<u32>,
    tree: &'a IntAVLTree<S, T>,
}

impl<'a, S, T> IntAVLTreeRefIterator<'a, S, T>
where
    S: IntAVLTreeStore<T>,
    T: Copy,
{
    fn new(tree: &'a IntAVLTree<S, T>) -> Self {
        let first_node = node_id_to_option(tree.root).and_then(|node| tree.first(node));
        IntAVLTreeRefIterator {
            current_node: first_node,
            tree: tree,
        }
    }
}

impl<'a, S, T> Iterator for IntAVLTreeRefIterator<'a, S, T>
where
    S: IntAVLTreeStore<T>,
    T: Copy,
{
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let result = self.current_node;
        if let Some(current_node) = self.current_node {
            self.current_node = self.tree.next(current_node);
        }
        result
    }
}

#[cfg(test)]
mod test {
    use crate::t_digest::avl_t_digest::aggregate_centroid::AggregateCentroid;
    use crate::t_digest::avl_t_digest::int_avl_tree::IntAVLTree;
    use crate::t_digest::avl_t_digest::int_avl_tree_store::IntAVLTreeStore;
    use crate::t_digest::avl_t_digest::tree_centroid_store::TreeCentroidStore;
    use crate::util::gen_uniform_centroid_random_weight_vec;

    #[test]
    fn add_uniform_centroids_and_find() {
        let mut centroids = gen_uniform_centroid_random_weight_vec::<f32>(1001);
        let mut tree: IntAVLTree<TreeCentroidStore<f32>, AggregateCentroid<f32>> =
            IntAVLTree::default();

        for centroid in &centroids {
            let agg_centroid = AggregateCentroid::from(centroid.clone());
            tree.add_by(agg_centroid);
        }

        centroids.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());

        for centroid in centroids {
            let agg_centroid = AggregateCentroid::from(centroid);
            let real_agg_centroid = tree.find_by(agg_centroid);
            assert_eq!(centroid.mean, real_agg_centroid.unwrap().centroid.mean);
            assert_eq!(centroid.weight, real_agg_centroid.unwrap().centroid.weight);
        }
    }

    #[test]
    fn add_uniform_centroids_and_iterator() {
        let mut centroids = gen_uniform_centroid_random_weight_vec::<f32>(1001);
        let mut tree: IntAVLTree<TreeCentroidStore<f32>, AggregateCentroid<f32>> =
            IntAVLTree::default();

        for centroid in &centroids {
            let agg_centroid = AggregateCentroid::from(centroid.clone());
            tree.add_by(agg_centroid);
        }

        centroids.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        let mut iter = tree.into_iter();
        for centroid in centroids {
            let node_id = iter.next().unwrap();
            let tree_centroid = tree.get_store().read(node_id);
            assert_eq!(centroid.mean, tree_centroid.centroid.mean);
            assert_eq!(centroid.weight, tree_centroid.centroid.weight);
        }
    }
}
