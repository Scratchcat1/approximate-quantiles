pub mod aggregate_centroid;
pub mod avl_tree_digest;
pub mod int_avl_tree;
pub mod int_avl_tree_store;
pub mod node_allocator;
pub mod tree_centroid_store;
// Derived from the AVL Tree Digest from https://github.com/tdunning/t-digest

/// Indicates empty node without the overhead of Some/None
pub const NIL: u32 = 0;

/// Converts a raw node id into an option
/// None if node == NIL
/// Some(node) otherwise
pub fn node_id_to_option(node: u32) -> Option<u32> {
    if node != NIL {
        Some(node)
    } else {
        None
    }
}
