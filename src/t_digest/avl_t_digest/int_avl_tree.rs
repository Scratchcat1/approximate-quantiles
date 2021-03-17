use crate::t_digest::avl_t_digest::node_allocator::NodeAllocator;
use crate::t_digest::avl_t_digest::NIL;
use std::cmp::Ordering;
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct IntAVLTree<T>
where
    T: PartialOrd + Copy + Add + Default,
{
    node_allocator: NodeAllocator,
    root: u32,
    parent: Vec<u32>,
    left: Vec<u32>,
    right: Vec<u32>,
    depth: Vec<u8>,
    data: Vec<T>,
}

impl<T> Default for IntAVLTree<T>
where
    T: PartialOrd + Copy + Add + Default,
{
    fn default() -> Self {
        Self::new(16)
    }
}

impl<T> IntAVLTree<T>
where
    T: PartialOrd + Copy + Add + Default,
{
    pub fn new(capacity: usize) -> Self {
        IntAVLTree {
            node_allocator: NodeAllocator::default(),
            root: NIL,
            parent: Vec::with_capacity(capacity),
            left: Vec::with_capacity(capacity),
            right: Vec::with_capacity(capacity),
            depth: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }

    /// Resize all arrays to the `new_capacity`
    /// The extra node slots are filled with NIL
    /// The extra data slots are filled with the default value of T
    fn resize(&mut self, new_capacity: usize) {
        self.parent.resize(new_capacity, NIL);
        self.left.resize(new_capacity, NIL);
        self.right.resize(new_capacity, NIL);
        self.depth.resize(new_capacity, 0);
        self.data.resize(new_capacity, T::default());
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

    /// Return the left child of the node
    #[inline]
    pub fn get_left(&self, node: u32) -> u32 {
        self.left[node as usize]
    }

    /// Return the right child of the node
    #[inline]
    pub fn get_right(&self, node: u32) -> u32 {
        self.right[node as usize]
    }

    /// Return the depth of the node
    #[inline]
    pub fn get_depth(&self, node: u32) -> u8 {
        self.depth[node as usize]
    }

    /// Assign `item` as data for `node`
    #[inline]
    pub fn set_data(&mut self, node: u32, item: T) {
        self.data[node as usize] = item;
    }

    /// Return the data of the node
    #[inline]
    pub fn get_data(&self, node: u32) -> T {
        self.data[node as usize]
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
        return Some(node);
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
        return Some(node);
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

    /// Add an item to the tree
    pub fn add<F>(&mut self, item: T, cmp: F)
    where
        F: FnMut(T) -> Ordering,
    {
        if self.root == NIL {
            self.root = self.node_allocator.new_node();
            self.set_data(self.root, item);
            self.fix_depth(self.root);
        } else {
            let mut node = self.root;
            assert!(self.get_parent(self.root) == NIL);

            loop {
                let parent;
                match cmp(self.get_data(node)) {
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

            self.set_data(node, item);
        }
    }
}
