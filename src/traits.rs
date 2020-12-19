use num_traits::{cast::FromPrimitive, Float};

pub trait Digest<F>
where
    F: Float,
{
    /// Add an item to the digest
    ///
    /// # Arguments
    ///
    /// * `item` The item to add to the digest
    fn add(&mut self, item: F);

    /// Add a buffer to the digest
    ///
    /// # Arguments
    ///
    /// * `buffer` The buffer to merge into the digest
    fn add_buffer(&mut self, buffer: &[F]);

    /// Estimate the quantile of an item
    /// # Arguments
    /// * `value` Item to estimate the quantile of.
    fn est_quantile_at_value(&mut self, value: F) -> F;

    /// Estimate the value at a particular quantile
    /// # Arguments
    /// * `quantile` The quantile to estimate the value of. 0 <= `quantile` <= 1.
    fn est_value_at_quantile(&mut self, quantile: F) -> F;
}
