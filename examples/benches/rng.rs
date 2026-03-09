#![feature(portable_simd)]

use std::simd::Simd;
use divan::AllocProfiler;
use rand::Rng;
use crab_metrics::math::{SimdPcgrng, Pcgrng, RandomGen, Pcgrnginc, RandomDistribution};

fn main() {
    divan::main();
}

// #[global_allocator]
// static ALLOC: AllocProfiler = AllocProfiler::system();

const SIZE: usize = 1_000_000;

#[divan::bench]
fn rng(benceher: divan::Bencher) {
    benceher.with_inputs(|| {
        vec![0; SIZE]
    }).bench_refs(move |buf| {
        let mut rng = Pcgrng::new(0);
        RandomGen::<u32>::fill_random(&mut rng, buf.as_mut_slice());
    })
}

#[divan::bench]
fn rng_inc(benceher: divan::Bencher) {
    benceher.with_inputs(|| {
        vec![0; SIZE]
    }).bench_refs(move |buf| {
        let mut rng = Pcgrnginc::new(0, 23);
        RandomGen::<u32>::fill_random(&mut rng, buf.as_mut_slice());
    })
}

#[divan::bench]
fn rng_float(benceher: divan::Bencher) {
    benceher.with_inputs(|| {
        vec![0.0; SIZE]
    }).bench_refs(move |buf| {
        let mut rng = Pcgrng::new(0);
        RandomGen::<f32>::fill_random(&mut rng, buf.as_mut_slice());
    })
}

#[divan::bench]
fn simd_rng_16(benceher: divan::Bencher) {
    const LANES: usize = 16;
    let seed: [u64; LANES] = [1234234; LANES];
    // let seed: [u64; LANES] = [145, 2134334, 3452354, 14123151, 123412354, 123541, 1234, 2341234, 145, 2134334, 3452354, 14123151, 123412354, 123541, 1234, 2341234];
    // let mut rng = SimdPcgrng::new(Simd::from_array(seed));
    benceher.with_inputs(|| {
        vec![0; SIZE]
    }).bench_refs(move |buf| {
        let mut rng = SimdPcgrng::new(Simd::from_array(seed));
        RandomGen::<Simd<u32, LANES>>::fill_random(&mut rng, buf.as_mut_slice());
    })
}

#[divan::bench]
fn simd_rng_8(benceher: divan::Bencher) {
    const LANES: usize = 8;
    let seed: [u64; LANES] = [1234234; LANES];
    // let seed: [u64; LANES] = [145, 2134334, 3452354, 14123151, 123412354, 123541, 1234, 2341234, 145, 2134334, 3452354, 14123151, 123412354, 123541, 1234, 2341234];
    // let mut rng = SimdPcgrng::new(Simd::from_array(seed));
    benceher.with_inputs(|| {
        vec![0; SIZE]
    }).bench_refs(move |buf| {
        let mut rng = SimdPcgrng::new(Simd::from_array(seed));
        RandomGen::<Simd<u32, LANES>>::fill_random(&mut rng, buf.as_mut_slice());
    })
}


#[divan::bench]
fn rng_float_simd_16(benceher: divan::Bencher) {
    const LANES: usize = 16;
    let mut seed: [u64; LANES] = [0; LANES];
    for i in 0..LANES {
        seed[i] = 53453453214 + i as u64 * 12341234;
    }
    benceher.with_inputs(|| {
        vec![0.0; SIZE]
    }).bench_refs(move |buf| {
        let mut rng = SimdPcgrng::new(Simd::from_array(seed));
        RandomGen::<Simd<f32, LANES>>::fill_random(&mut rng, buf.as_mut_slice());
    })
}



// #[divan::bench]
// fn sample_normal(benceher: divan::Bencher) {
//
// }
