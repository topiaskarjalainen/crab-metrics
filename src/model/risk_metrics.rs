use crate::math::{Pcgrng, RandomGen, SimdPcgrng};
use crate::util::{AssignFromArray, NaiveAssign, SimdAssign};
use std::marker::PhantomData;
use std::simd::{Simd, StdFloat};

pub struct SimulationData<'a, const N: usize, const R: usize>
where
    [(); N * N]:,
    [(); R * R]:,
{
    pub cpty_ids: Vec<u32>,
    pub cpty_ind: Vec<u8>,
    pub ead: Vec<f32>,
    pub lgd: Vec<f32>,
    pub pd: Vec<f32>,
    pub r_factor: Vec<f32>,
    pub corr_cholesky: &'a [f32; N * N],
    pub transition_matrix: &'a [f32; R * R],
    pub master_scale: &'a [f32; R],
}

impl<'a, const N: usize, const R: usize> SimulationData<'a, N, R>
where
    [(); N * N]:,
    [(); R * R]:,
{
    pub fn verify_industry_indices(&self) {
        let max_ind = *self.cpty_ind.iter().max().unwrap();
        assert!(
            max_ind <= N as u8,
            "Industry index out of bounds for correlation matrix"
        );
    }
}

pub struct RiskMetricsResult {
    pub var: f32,
    pub cvar: f32,
}

pub struct CptyLevelResults {
    pub cpty_id: Vec<u32>,
    pub var: Vec<f32>,
    pub cvar: Vec<f32>,
}

pub trait SimFunctions {
    type ValType;
    type RNG: RandomGen<Self::ValType>;

    /// The normally distributed systematic shocks for all industries, correlated according to the correlation matrix.
    fn get_systematic_shocks<const N: usize>(rng: &mut Self::RNG, buf: &mut [f32]);
    /// The normally distributed idiosynchratic shocks for all counterparties.
    fn get_idiosynchratic_shocks(rng: &mut Self::RNG, buf: &mut [f32]);
    /// Assign for all in transition_values the index of the first larger or equal value in transition_thresholds,
    fn assign_pd<const PD_LOOKUP_SIZE: usize>(
        transition_thresholds: &[f32; PD_LOOKUP_SIZE],
        transition_values: &[f32],
        out: &mut [u8],
    );

    fn calculate_thresholds(sync_shocks: &[f32], idio_shocks: &[f32], r: &[f32], out: &mut [f32]);
}

pub struct BaseSimFunctions;
pub struct SimdSimFunctions<const LANES: usize>;

impl SimFunctions for BaseSimFunctions {
    type ValType = f32;
    type RNG = Pcgrng;

    fn get_systematic_shocks<const N: usize>(rng: &mut Self::RNG, buf: &mut [f32]) {
        todo!()
    }

    fn get_idiosynchratic_shocks(rng: &mut Self::RNG, buf: &mut [f32]) {
        todo!()
    }

    #[inline(always)]
    fn assign_pd<const PD_LOOKUP_SIZE: usize>(
        transition_thresholds: &[f32; PD_LOOKUP_SIZE],
        transition_values: &[f32],
        out: &mut [u8],
    ) {
        NaiveAssign::assign_from_array(transition_thresholds, transition_values, out);
    }

    fn calculate_thresholds(sync_shocks: &[f32], idio_shocks: &[f32], r: &[f32], out: &mut [f32]) {
        sync_shocks
            .iter()
            .zip(idio_shocks.iter())
            .zip(r.iter())
            .zip(out.iter_mut())
            .for_each(|a| {
                let (((&sync, &idio), &r), out) = a;
                *out = r * idio + (1.0 - r.powi(2)).sqrt() * sync;
            })
    }
}

impl<const LANES: usize> SimFunctions for SimdSimFunctions<LANES> {
    type ValType = Simd<f32, LANES>;
    type RNG = SimdPcgrng<LANES>;

    fn get_systematic_shocks<const N: usize>(rng: &mut Self::RNG, buf: &mut [f32]) {
        todo!()
    }

    fn get_idiosynchratic_shocks(rng: &mut Self::RNG, buf: &mut [f32]) {
        todo!()
    }

    #[inline(always)]
    fn assign_pd<const PD_LOOKUP_SIZE: usize>(
        transition_thresholds: &[f32; PD_LOOKUP_SIZE],
        transition_values: &[f32],
        out: &mut [u8],
    ) {
        SimdAssign::<LANES>::assign_from_array(transition_thresholds, transition_values, out);
    }

    fn calculate_thresholds(sync_shocks: &[f32], idio_shocks: &[f32], r: &[f32], out: &mut [f32]) {
        // type Vx = Simd<f32, LANES>;

        let (sync_chunks, sync_rem) = sync_shocks.as_chunks::<LANES>();
        let (idio_chunks, idio_rem) = idio_shocks.as_chunks::<LANES>();
        let (r_chunks, r_rem) = r.as_chunks::<LANES>();
        let (out_chunks, out_rem) = out.as_chunks_mut::<LANES>();

        for (((sync_chunk, idio_chunk), r_chunk), out_chunk) in sync_chunks
            .iter()
            .zip(idio_chunks.iter())
            .zip(r_chunks.iter())
            .zip(out_chunks.iter_mut())
        {
            let v_sync = Simd::<f32, LANES>::from_slice(sync_chunk);
            let v_idio = Simd::<f32, LANES>::from_slice(idio_chunk);
            let v_r = Simd::<f32, LANES>::from_slice(r_chunk);

            let one = Simd::splat(1.0);
            let v_out = v_r * v_idio + (one - v_r * v_r).sqrt() * v_sync;
            out_chunk.copy_from_slice(&v_out.to_array());
        }
        // Handle remainder with calling the default implementation for the remaining elements
        if !sync_rem.is_empty() {
            BaseSimFunctions::calculate_thresholds(sync_rem, idio_rem, r_rem, out_rem);
        }
    }
}

pub struct RiskMetrics<Sim: SimFunctions> {
    pub confidence_level: f32,
    pub n_simulations: usize,
    pub rng: Sim::RNG,
    pub _sim: PhantomData<Sim>,
}

impl<Sim: SimFunctions> RiskMetrics<Sim> {
    pub fn simulate_portfolio_level<const N_INDUSTRIES: usize, const R: usize>(
        &mut self,
        data: &SimulationData<N_INDUSTRIES, R>,
    ) -> RiskMetricsResult
    where
        [(); N_INDUSTRIES * N_INDUSTRIES]:,
        [(); R * R]:,
    {
        data.verify_industry_indices(); // Guards against invalid industry indices -> can use unchecked access in the simulation loop
        let mut idiosyncratic_shocks = vec![0f32; data.cpty_ids.len()];
        let mut transition_values = vec![0f32; data.cpty_ids.len()];

        let systematic_shocks = &mut [0f32; N_INDUSTRIES];
        let mut systematic_shocks_by_cpty = vec![0f32; data.cpty_ids.len()];
        let rng = &mut self.rng;
        // Main loop
        for _ in 0..self.n_simulations {
            Sim::get_systematic_shocks::<N_INDUSTRIES>(rng, systematic_shocks);
            Sim::get_idiosynchratic_shocks(rng, &mut idiosyncratic_shocks);
            Sim::assign_pd(
                &data.master_scale,
                &transition_values,
                &mut systematic_shocks_by_cpty,
            );

            // Calc factors
            Sim::calculate_thresholds(
                systematic_shocks,
                &idiosyncratic_shocks,
                &data.r_factor,
                &mut transition_values,
            ););
        }

        todo!()
    }
}
