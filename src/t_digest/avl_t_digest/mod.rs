pub mod int_avl_tree;
pub mod node_allocator;
// Derived from the AVL Tree Digest from https://github.com/tdunning/t-digest

/// Indicates empty node without the overhead of Some/None
pub const NIL: u32 = 0;
