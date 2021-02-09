use num_traits::Float;
use std::f64::consts::PI;

pub fn k0<F>(quantile: F, comp_factor: F, _: F) -> F
where
    F: Float,
{
    (quantile * comp_factor) / F::from(2.0).unwrap()
}

pub fn inv_k0<F>(scale: F, comp_factor: F, _: F) -> F
where
    F: Float,
{
    (scale * F::from(2.0).unwrap()) / comp_factor
}

pub fn k1<F>(quantile: F, comp_factor: F, _: F) -> F
where
    F: Float,
{
    (comp_factor / (F::from(2.0).unwrap() * F::from(PI).unwrap()))
        * (F::from(2.0).unwrap() * quantile - F::from(1.0).unwrap()).asin()
}

pub fn inv_k1<F>(scale: F, comp_factor: F, _: F) -> F
where
    F: Float,
{
    (F::from(1.0).unwrap()
        + (F::from(2.0).unwrap() * F::from(PI).unwrap() * scale / comp_factor).sin())
        / F::from(2.0).unwrap()
}

pub fn k2<F>(quantile: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    (comp_factor / (F::from(4.0).unwrap() * (n / comp_factor).log10() + F::from(24.0).unwrap()))
        * (quantile / (F::from(1.0).unwrap() - quantile)).log10()
}

pub fn inv_k2<F>(scale: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    let x = F::from(10.0).unwrap().powf(
        (scale * (F::from(4.0).unwrap() * (n / comp_factor).log10() + F::from(24.0).unwrap()))
            / comp_factor,
    );
    return x / (F::from(1.0).unwrap() + x);
}

pub fn k2_asym<F>(quantile: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    let effective_quantile = quantile * F::from(0.5).unwrap();
    k2(effective_quantile, comp_factor, n)
}

pub fn inv_k2_asym<F>(scale: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    inv_k2(scale, comp_factor, n) * F::from(2.0).unwrap()
}

pub fn k2n<F>(quantile: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    let mod_comp_factor =
        comp_factor / F::from(10.0).unwrap() * n.log10().powf(F::from(2.0).unwrap());
    (mod_comp_factor
        / (F::from(4.0).unwrap() * (n / mod_comp_factor).log10() + F::from(24.0).unwrap()))
        * (quantile / (F::from(1.0).unwrap() - quantile)).log10()
}

pub fn inv_k2n<F>(scale: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    let mod_comp_factor =
        comp_factor / F::from(10.0).unwrap() * n.log10().powf(F::from(2.0).unwrap());
    let x = F::from(10.0).unwrap().powf(
        (scale * (F::from(4.0).unwrap() * (n / mod_comp_factor).log10() + F::from(24.0).unwrap()))
            / mod_comp_factor,
    );
    return x / (F::from(1.0).unwrap() + x);
}

pub fn k3<F>(quantile: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    let factor = match quantile <= F::from(0.5).unwrap() {
        true => (F::from(2.0).unwrap() * quantile).log10(),
        false => -(F::from(2.0).unwrap() * (F::from(1.0).unwrap() - quantile)).log10(),
    };
    (comp_factor / (F::from(4.0).unwrap() * (n / comp_factor).log10() + F::from(21.0).unwrap()))
        * factor
}

pub fn inv_k3<F>(scale: F, comp_factor: F, n: F) -> F
where
    F: Float,
{
    let pow = (scale
        * (F::from(4.0).unwrap() * (n / comp_factor).log10() + F::from(21.0).unwrap()))
        / comp_factor;

    let q_low = F::from(10.0).unwrap().powf(pow) / F::from(2.0).unwrap();
    let q_high =
        (F::from(2.0).unwrap() - F::from(10.0).unwrap().powf(-pow)) / F::from(2.0).unwrap();
    match (F::from(0.5).unwrap() - q_low).abs() > (F::from(0.5).unwrap() - q_high).abs() {
        true => q_high,
        false => q_low,
    }
}

#[cfg(test)]
mod scale_functions_test {
    use crate::t_digest::scale_functions::{
        inv_k0, inv_k1, inv_k2, inv_k2_asym, inv_k2n, inv_k3, k0, k1, k2, k2_asym, k2n, k3,
    };
    use approx::assert_relative_eq;

    #[test]
    fn k0_properties() {
        assert_relative_eq!(k0(0.0, 10.0, 0.0), 0.0);
    }

    #[test]
    fn inv_k0_properties() {
        for i in 0..100 {
            assert_relative_eq!(inv_k0(k0(i as f64, 10.0, 0.0), 10.0, 0.0), i as f64);
        }
    }

    #[test]
    fn k1_properties() {
        assert_relative_eq!(k1(1.0, 10.0, 0.0), 10.0 / 4.0);
    }

    #[test]
    fn inv_k1_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k1(k1(q, 10.0, 0.0), 10.0, 0.0), q);
        }
    }

    #[test]
    fn inv_k2_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k2(k2(q, 10.0, 10.0), 10.0, 10.0), q);
        }
    }

    #[test]
    fn inv_k2_asym_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k2_asym(k2_asym(q, 10.0, 10.0), 10.0, 10.0), q);
        }
    }

    #[test]
    fn inv_k2n_properties() {
        for i in 0..100 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k2n(k2n(q, 10.0, 10.0), 10.0, 10.0), q);
        }
    }

    #[test]
    fn inv_k3_properties() {
        for i in 1..99 {
            let q = i as f64 / 100.0;
            assert_relative_eq!(inv_k3(k3(q, 10.0, 10.0), 10.0, 10.0), q, epsilon = 0.01);
        }
    }
}
