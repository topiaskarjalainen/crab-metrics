use std::simd::prelude::SimdFloat;
use std::simd::{Simd, StdFloat};

pub trait SpecialFunctions<T> {
    type InnerType;

    fn exp(x: T) -> T;
    fn ln(x: T) -> T;
    fn erf(x: T) -> T;

    fn apply_inplace<F: Fn(T) -> T>(buf: &mut [Self::InnerType], f: F);
}


pub struct BaseMath;

impl BaseMath {
    #[inline(always)]
    fn _erf_positive(x: f32) -> f32 {
        const A1: f32 = 0.254829592;
        const A2: f32 = -0.284496736;
        const A3: f32 = 1.421413741;
        const A4: f32 = -1.453152027;
        const A5: f32 = 1.061405429;
        const P: f32= 0.3275911;

        let t = 1.0 / (1.0 + P * x);
        1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp()
    }
}

impl SpecialFunctions<f32> for BaseMath {
    type InnerType = f32;

    #[inline(always)]
    fn exp(x: f32) -> f32 {
        x.exp()
    }

    #[inline(always)]
    fn ln(x: f32) -> f32 {
        x.ln()
    }

    fn erf(x: f32) -> f32 {
        // Abramowitz and Stegun formula 7.1.26
        let sign = x.signum();
        let val = Self::_erf_positive(x);
        sign * val
    }

    fn apply_inplace<F: Fn(f32) -> f32>(buf: &mut [Self::InnerType], f: F) {
        for x in buf.iter_mut() {
            *x = f(*x);
        }
    }
}

pub struct VecMath<const LANES: usize>;

impl<const LANES: usize> VecMath<LANES> {
    const M1: Simd<f32, LANES> = Simd::splat(0.254829592);
}

impl<const LANES: usize> SpecialFunctions<Simd<f32, LANES>> for VecMath<LANES> {
    type InnerType = f32;

    #[inline(always)]
    fn exp(x: Simd<f32, LANES>) -> Simd<f32, LANES> {
        x.exp()
    }

    #[inline(always)]
    fn ln(x: Simd<f32, LANES>) -> Simd<f32, LANES> {
        x.ln()
    }

    fn erf(x: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let sign = x.signum();
        let t = Simd::splat(1.0) / (Simd::splat(1.0) + Self::M1 * x);
        let val = Simd::splat(1.0) - (((((Simd::splat(1.061405429) * t + Simd::splat(-1.453152027)) * t) + Simd::splat(1.421413741)) * t + Simd::splat(-0.284496736)) * t + Self::M1) * t * (-x * x).exp();
        sign * val
    }

    fn apply_inplace<F: Fn(Simd<f32, LANES>) -> Simd<f32, LANES>>(buf: &mut [Self::InnerType], f: F) {
        let (head, tail) = buf.as_chunks_mut::<LANES>();
        for chunk in head {
            let vec = Simd::from_slice(chunk);
            let res = f(vec);
            chunk.copy_from_slice(&res.to_array());
        }
        // remainder, make a simd and fill with dummy at the end, then apply and take the first few
        let mut dummy = [0.0; LANES];
        dummy[..tail.len()].copy_from_slice(tail);
        let vec = Simd::from_array(dummy);
        let res = f(vec);
        tail.copy_from_slice(&res.to_array()[..tail.len()]);
    }
}



// pub struct BufferMath<T: SpecialFunctions<T>>(std::marker::PhantomData<T>);



