#![feature(portable_simd)]

use crab_metrics::math::{
    BaseMath, Pcgrng, Pcgrnginc, RandomDistribution, RandomGen, SimdPcgrng, SpecialFunctions,
    VecMath,
};
use crab_metrics::util;
use crab_metrics::util::AssignFromArray;
use divan::{AllocProfiler, Bencher};
use rand::Rng;
use std::simd::Simd;

fn main() {
    divan::main();
}

const SIZE: usize = 1_000_000;

#[divan::bench]
fn erf(bencher: Bencher) {
    bencher
        .with_inputs(|| vec![0.0; SIZE])
        .bench_refs(move |buf| {
            BaseMath::apply_inplace(buf.as_mut_slice(), BaseMath::erf);
        })
}

const LANES: usize = 16;

#[divan::bench]
fn erf_vec(bencher: Bencher) {
    bencher
        .with_inputs(|| vec![0.0; SIZE])
        .bench_refs(move |buf| {
            VecMath::<LANES>::apply_inplace(buf.as_mut_slice(), VecMath::erf);
        })
}

#[divan::bench]
fn lookup(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let look_up = [
                0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
                2.2, //
                2.3, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ];
            let elems = vec![0.5; N_CPTY];
            let mut out = vec![0u8; N_CPTY];
            (look_up, elems, out)
        })
        .bench_refs(move |(lk, el, out)| {
            util::StdAssign::assign_from_array(lk, el, out);
        });
}

const N_CPTY: usize = 50_000;

#[divan::bench]
fn lookup_naive(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let look_up = [
                0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
                2.2, //
                2.3, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ];
            let elems = vec![0.5; N_CPTY];
            let mut out = vec![0u8; N_CPTY];
            (look_up, elems, out)
        })
        .bench_refs(move |(lk, el, out)| {
            util::NaiveAssign::assign_from_array(lk, el, out);
        });
}

#[divan::bench]
fn lookup_simd_16(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let look_up = [
                0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
                2.2, //
                2.3, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ];
            let elems = vec![0.5; N_CPTY];
            let mut out = vec![0u8; N_CPTY];
            (look_up, elems, out)
        })
        .bench_refs(move |(lk, el, out)| {
            util::SimdAssign::<16>::assign_from_array(lk, el, out);
        });
}
