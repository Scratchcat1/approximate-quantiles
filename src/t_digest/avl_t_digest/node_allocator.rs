use crate::t_digest::avl_t_digest::NIL;

/// Allocates ids either by creating a new id or using a released/marked as unused id
#[derive(Clone, Debug)]
pub struct NodeAllocator {
    /// The next node id which will be allocated.
    next_node: u32,
    /// Vector of released nodes. Used as a stack.
    released_nodes: Vec<u32>,
}

impl NodeAllocator {
    pub fn new() -> Self {
        NodeAllocator {
            next_node: NIL + 1,
            released_nodes: Vec::new(),
        }
    }

    /// Returns a new node
    /// Either generates a new value or uses a released node
    pub fn new_node(&mut self) -> u32 {
        match self.released_nodes.pop() {
            Some(node) => node,
            None => {
                assert!(self.next_node < u32::MAX);
                let node = self.next_node;
                self.next_node += 1;
                node
            }
        }
    }

    /// Mark a node for reuse
    /// # Arguments
    /// `node` The node to release
    pub fn release(&mut self, node: u32) {
        assert!(node < self.next_node);
        self.released_nodes.push(node);
    }

    pub fn size(&self) -> u32 {
        return self.next_node - self.released_nodes.len() as u32 - 1;
    }
}

impl Default for NodeAllocator {
    fn default() -> Self {
        Self::new()
    }
}
