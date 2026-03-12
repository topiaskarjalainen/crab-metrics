use divan::Bencher;

pub fn main() {
    divan::main();
}

mod factors_calculations {
    use crab_metrics::model::risk_metrics as rm;
    use crab_metrics::model::risk_metrics::SimFunctions;
    use divan::Bencher;

    fn get_args_for_factors(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let a = vec![0.0; n];
        let b = vec![0.0; n];
        let c = vec![0.0; n];
        (a, b, c)
    }

    #[divan::bench]
    fn base(bencher: Bencher) {
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
    fn simd_16(bencher: Bencher) {
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
    fn simd_32(bencher: Bencher) {
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
}

#[divan::bench_group]
mod loss_calculations {
    use crab_metrics::model::risk_metrics as rm;
    use crab_metrics::model::risk_metrics::SimFunctions;
    use divan::Bencher;
    use divan::__private::Arg;

    struct Args {
        pd: Vec<f32>,
        lgd: Vec<f32>,
        ead: Vec<f32>,
        m: Vec<f32>,
    }

    fn get_args_for_loss_calculations(n: usize) -> Args {
        let pd = vec![0.0; n];
        let lgd = vec![0.0; n];
        let ead = vec![0.0; n];
        let m = vec![0.0; n];
        Args { pd, lgd, ead, m }
    }

    const TRESHOLD: f32 = 0.5;
    const N: usize = 1_000_000;

    #[divan::bench]
    fn base(bencher: Bencher) {
        bencher
            .with_inputs(|| get_args_for_loss_calculations(N))
            .bench_refs(|args| {
                let mut out = 0.0;

                rm::BaseSimFunctions::default_loss(
                    args.pd.as_slice(),
                    args.lgd.as_slice(),
                    args.ead.as_slice(),
                    args.m.as_slice(),
                    TRESHOLD,
                    &mut out,
                );

                divan::black_box(out);
            })
    }

    #[divan::bench]
    fn simd_16(bencher: Bencher) {
        bencher
            .with_inputs(|| get_args_for_loss_calculations(N))
            .bench_refs(|args| {
                let mut out = 0.0;

                rm::SimdSimFunctions::<16>::default_loss(
                    args.pd.as_slice(),
                    args.lgd.as_slice(),
                    args.ead.as_slice(),
                    args.m.as_slice(),
                    TRESHOLD,
                    &mut out,
                );

                divan::black_box(out);
            })
    }

    #[divan::bench]
    fn simd_32(bencher: Bencher) {
        bencher
            .with_inputs(|| get_args_for_loss_calculations(N))
            .bench_refs(|args| {
                let mut out = 0.0;

                rm::SimdSimFunctions::<32>::default_loss(
                    args.pd.as_slice(),
                    args.lgd.as_slice(),
                    args.ead.as_slice(),
                    args.m.as_slice(),
                    TRESHOLD,
                    &mut out,
                );

                divan::black_box(out);
            })
    }
}
