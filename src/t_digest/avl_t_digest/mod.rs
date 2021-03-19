pub mod aggregate_centroid;
pub mod c_sum;
pub mod int_avl_tree;
pub mod node_allocator;
pub mod tree_aggregate;
// Derived from the AVL Tree Digest from https://github.com/tdunning/t-digest

/// Indicates empty node without the overhead of Some/None
pub const NIL: u32 = 0;
