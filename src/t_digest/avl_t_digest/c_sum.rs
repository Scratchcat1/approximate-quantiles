use std::cmp::Ordering;

pub trait CSum<T> {
    fn head_sum<F>(&self, cmp: F) -> u32
    where
        F: FnMut(T) -> Ordering;
}
