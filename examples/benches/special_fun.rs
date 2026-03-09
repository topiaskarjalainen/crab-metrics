#![feature(portable_simd)]

use std::simd::Simd;
use divan::{AllocProfiler, Bencher};
use rand::Rng;
use crab_metrics::math::{SimdPcgrng, Pcgrng, RandomGen, Pcgrnginc, RandomDistribution, BaseMath, SpecialFunctions, VecMath};


fn main() {
    divan::main();
}

const SIZE: usize = 1_000_000;

#[divan::bench]
fn erf(bencher: Bencher) {
    bencher.with_inputs(|| {
        vec![0.0; SIZE]
    }).bench_refs(move |buf| {
        BaseMath::apply_inplace(buf.as_mut_slice(), BaseMath::erf);
    })
}

const LANES: usize = 16;

#[divan::bench]
fn erf_vec(bencher: Bencher) {
    bencher.with_inputs(|| {
        vec![0.0; SIZE]
    }).bench_refs(move |buf| {
        VecMath::<LANES>::apply_inplace(buf.as_mut_slice(), VecMath::erf);
    })
}

