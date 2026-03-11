use divan::Bencher;

use crab_metrics::model::risk_metrics as rm;
use crab_metrics::model::risk_metrics::SimFunctions;

pub fn main() {
    divan::main();
}

fn get_args_for_factors(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a = vec![0.0; n];
    let b = vec![0.0; n];
    let c = vec![0.0; n];
    (a, b, c)
}

#[divan::bench]
fn calc_factors(bencher: Bencher) {
    bencher
        .with_inputs(|| get_args_for_factors(1_000_000))
        .bench_refs(|(a, b, c)| {
            let mut out = vec![0.0; a.len()];
            rm::BaseSimFunctions::calculate_thresholds(
                a.as_slice(),
                b.as_slice(),
                c.as_slice(),
                out.as_mut_slice(),
            );
        })
}

#[divan::bench]
fn calc_factors_simd_16(bencher: Bencher) {
    bencher
        .with_inputs(|| get_args_for_factors(1_000_000))
        .bench_refs(|(a, b, c)| {
            let mut out = vec![0.0; a.len()];
            const LANES: usize = 16;
            rm::SimdSimFunctions::<LANES>::calculate_thresholds(
                a.as_slice(),
                b.as_slice(),
                c.as_slice(),
                out.as_mut_slice(),
            );
        })
}

#[divan::bench]
fn calc_factors_simd_32(bencher: Bencher) {
    bencher
        .with_inputs(|| get_args_for_factors(1_000_000))
        .bench_refs(|(a, b, c)| {
            let mut out = vec![0.0; a.len()];
            const LANES: usize = 32;
            rm::SimdSimFunctions::<LANES>::calculate_thresholds(
                a.as_slice(),
                b.as_slice(),
                c.as_slice(),
                out.as_mut_slice(),
            );
        })
}
