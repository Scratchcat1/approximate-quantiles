use std::f64::consts::PI;

pub fn k0(quantile: f64, comp_factor: f64) -> f64 {
    (quantile * comp_factor) / 2.0
}

pub fn inv_k0(scale: f64, comp_factor: f64) -> f64 {
    (scale * 2.0) / comp_factor
}

pub fn k1(quantile: f64, comp_factor: f64) -> f64 {
    (comp_factor / (2.0 * PI)) * (2.0 * quantile - 1.0).asin()
}

pub fn inv_k1(scale: f64, comp_factor: f64) -> f64 {
    (1.0 + (2.0 * PI * scale / comp_factor).sin()) / 2.0
}

pub fn k2(quantile: f64, comp_factor: f64) -> f64 {
    let n = 10.0;
    (comp_factor / (4.0 * (n / comp_factor).log10() + 24.0)) * (quantile / (1.0 - quantile)).log10()
}

pub fn inv_k2(scale: f64, comp_factor: f64) -> f64 {
    let n: f64 = 10.0;
    let x = (10.0 as f64).powf((scale * (4.0 * (n / comp_factor).log10() + 24.0)) / comp_factor);
    return x / (1.0 + x);
}

pub fn k3(quantile: f64, comp_factor: f64) -> f64 {
    let n = 10.0;
    let factor = match quantile <= 0.5 {
        true => (2.0 * quantile).log10(),
        false => -(2.0 * (1.0 - quantile)).log10(),
    };
    (comp_factor / (4.0 * (n / comp_factor).log10() + 21.0)) * factor
}

pub fn inv_k3(scale: f64, comp_factor: f64) -> f64 {
    let n = 10.0;
    let pow = (scale * (4.0 * (n / comp_factor).log10() + 21.0)) / comp_factor;

    let q_low = (10.0 as f64).powf(pow) / 2.0;
    let q_high = (2.0 - (10.0 as f64).powf(-pow)) / 2.0;
    match (0.5 - q_low).abs() > (0.5 - q_high).abs() {
        true => q_high,
        false => q_low,
    }
}
