use std::cmp::Ordering;

pub trait IntAVLTreeStore<T> {
    /// Create a new store with capacity `capacity`
    fn with_capacity(capacity: u32) -> Self;

    /// Resize the underlying store as the tree is resizing
    fn resize(&mut self, new_capacity: u32);

    /// Merge an item with a node in the store
    fn merge(&mut self, node: u32, item: T);

    /// Copy an item into the node
    fn copy(&mut self, node: u32, item: T);

    /// Read the data associated with the node
    fn read(&self, node: u32) -> T;

    /// Compare `item` to the data store at `node`
    /// Comparison ordering is item {ord} data[node]
    fn compare(&self, node: u32, item: T) -> Ordering;

    /// Fix aggregates
    /// Used when store keeps track of values derived from each child e.g. A summation tree
    /// Children may be NIL
    fn fix_aggregates(&mut self, node: u32, left_child: u32, right_child: u32);
}
