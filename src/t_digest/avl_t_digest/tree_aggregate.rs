/// Trait for things which have an attribute which should be an aggregate of it's children in a tree
pub trait TreeAggregate<T> {
    /// Update the current node's aggregate because one of the children was updated.
    fn fix_aggregate(&mut self, left_child: Option<&T>, right_child: Option<&T>);
}
