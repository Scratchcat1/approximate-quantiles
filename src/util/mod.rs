pub mod linear_digest;

pub fn weighted_average(x1: f64, w1: f64, x2: f64, w2: f64) -> f64 {
    let weighted = (x1 * w1 + x2 * w2) / (w1 + w2);
    let max = f64::max(x1, x2);
    let min = f64::min(x1, x2);
    f64::max(min, f64::min(weighted, max))
}
