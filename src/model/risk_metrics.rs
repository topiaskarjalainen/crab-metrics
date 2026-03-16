use crate::math::{Pcgrng, RandomDistribution, RandomGen, SimdPcgrng, STD_NORMAL};
use crate::util::{AssignFromArray, NaiveAssign, SimdAssign};
use std::marker::PhantomData;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::{Mask, Select, Simd, StdFloat};

pub struct StaticSimulationData<'a, const N: usize, const R: usize>
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

impl<'a, const N: usize, const R: usize> StaticSimulationData<'a, N, R>
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

    /// calculate total default EL for all counterparties based on the assigned PD and the EAD and LGD values
    fn default_loss(
        pd: &[f32],
        ead: &[f32],
        lgd: &[f32],
        maturity_adjustment: &[f32],
        default_threshold: f32,
        out: &mut f32,
    );
    /// Calculate total migration EL for all counterparties based on the assigned PD and the EAD and LGD values and the transition matrix
    fn migration_loss(pd: &[f32], ead: &[f32], lgd: &[f32], default_threshold: f32, out: &mut f32);

    fn calc_losses(
        pd: &[f32],
        ead: &[f32],
        lgd: &[f32],
        maturity_adjustment: &[f32],
        default_threshold: f32,
        out_default_loss: &mut f32,
        out_migration_loss: &mut f32,
    );
}

pub struct BaseSimFunctions;
pub struct SimdSimFunctions<const LANES: usize>;

impl SimFunctions for BaseSimFunctions {
    type ValType = f32;
    type RNG = Pcgrng;

    fn get_systematic_shocks<const N: usize>(rng: &mut Self::RNG, buf: &mut [f32]) {
        // Drwaw the shocks and assign to the correct industry according to the ind_buf
        STD_NORMAL.sample_into(rng, buf);
    }

    fn get_idiosynchratic_shocks(rng: &mut Self::RNG, buf: &mut [f32]) {
        // Drwaw the shocks and assign to the correct industry according to the ind_buf
        STD_NORMAL.sample_into(rng, buf);
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

    #[inline]
    fn default_loss(
        pd: &[f32],
        ead: &[f32],
        lgd: &[f32],
        maturity_adjustment: &[f32],
        default_threshold: f32,
        out: &mut f32,
    ) {
        let sum = pd
            .iter()
            .zip(ead.iter())
            .zip(lgd.iter())
            .zip(maturity_adjustment.iter())
            .fold(0f32, |acc, a| {
                let (((&pd, &ead), &lgd), &ma) = a;
                acc + if pd >= default_threshold {
                    ead * lgd * ma
                } else {
                    0.0
                }
            });
        *out = sum;
    }

    fn migration_loss(pd: &[f32], ead: &[f32], lgd: &[f32], default_threshold: f32, out: &mut f32) {
        let sum = pd
            .iter()
            .zip(ead.iter())
            .zip(lgd.iter())
            .fold(0f32, |acc, a| {
                let ((&pd, &ead), &lgd) = a;
                acc + if pd < default_threshold {
                    ead * lgd * pd
                } else {
                    0.0
                }
            });
        *out = sum;
    }

    fn calc_losses(
        pd: &[f32],
        ead: &[f32],
        lgd: &[f32],
        maturity_adjustment: &[f32],
        default_threshold: f32,
        out_default_loss: &mut f32,
        out_migration_loss: &mut f32,
    ) {
        let mut default_loss_sum = 0f32;
        let mut migration_loss_sum = 0f32;
        pd.iter()
            .zip(ead.iter())
            .zip(lgd.iter())
            .zip(maturity_adjustment.iter())
            .for_each(|a| {
                let (((&pd, &ead), &lgd), &ma) = a;
                if pd >= default_threshold {
                    default_loss_sum += ead * lgd * ma;
                } else {
                    migration_loss_sum += ead * lgd * pd * ma;
                }
            });
        *out_default_loss = default_loss_sum;
        *out_migration_loss = migration_loss_sum;
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

    #[inline]
    fn default_loss(
        pd: &[f32],
        ead: &[f32],
        lgd: &[f32],
        maturity_adjustment: &[f32],
        default_threshold: f32,
        out: &mut f32,
    ) {
        type Vx<const L: usize> = Simd<f32, L>;
        type Mx<const L: usize> = Mask<i32, L>;

        let (pd_chunks, pd_rem) = pd.as_chunks::<LANES>();
        let (ead_chunks, ead_rem) = ead.as_chunks::<LANES>();
        let (lgd_chunks, lgd_rem) = lgd.as_chunks::<LANES>();
        let (ma_chunks, ma_rem) = maturity_adjustment.as_chunks::<LANES>();
        let mut sum = Vx::<LANES>::splat(0.0);
        let v_threshold = Vx::<LANES>::splat(default_threshold);
        pd_chunks
            .iter()
            .zip(ead_chunks.iter())
            .zip(lgd_chunks.iter())
            .zip(ma_chunks.iter())
            .for_each(|(((pd, ead), lgd), ma)| {
                let v_ead = Vx::<LANES>::from_slice(ead);
                let v_lgd = Vx::<LANES>::from_slice(lgd);
                let v_ma = Vx::<LANES>::from_slice(ma);
                let v_loss = v_ead * v_lgd * v_ma;

                let v_pd = Vx::<LANES>::from_slice(pd);
                let mask: Mx<LANES> = v_pd.simd_ge(v_threshold);

                sum += mask.select(v_loss, Vx::splat(0.0));
            });
        let mut total_sum = sum.reduce_sum();
        // Handle remainder with calling the default implementation for the remaining elements
        if !pd_rem.is_empty() {
            let mut rem_sum = 0f32;
            BaseSimFunctions::default_loss(
                pd_rem,
                ead_rem,
                lgd_rem,
                ma_rem,
                default_threshold,
                &mut rem_sum,
            );
            total_sum += rem_sum;
        }
        *out = total_sum;
    }

    fn migration_loss(pd: &[f32], ead: &[f32], lgd: &[f32], default_threshold: f32, out: &mut f32) {
        todo!()
    }

    fn calc_losses(
        pd: &[f32],
        ead: &[f32],
        lgd: &[f32],
        maturity_adjustment: &[f32],
        default_threshold: f32,
        out_default_loss: &mut f32,
        out_migration_loss: &mut f32,
    ) {
        todo!()
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
        data: &StaticSimulationData<N_INDUSTRIES, R>,
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
        let mut assigned_pd = vec![0u8; data.cpty_ids.len()];

        let mut dyn_data = DynamicSimulationData {
            systematic_shocks_by_cpty: &mut systematic_shocks_by_cpty,
            idiosynchratic_shocks_by_cpty: &mut idiosyncratic_shocks,
            transition_values: &mut transition_values,
            assigned_pd: &mut assigned_pd,
        };

        // Main loop
        for _ in 0..self.n_simulations {
            self.loop_fn(&data, &mut dyn_data);
        }
        todo!()
    }

    #[inline(always)]
    pub fn loop_fn<'a, const N_INDUSTRIES: usize, const R: usize>(
        &mut self,
        static_data: &StaticSimulationData<N_INDUSTRIES, R>,
        dyn_data: &mut DynamicSimulationData<'a>,
    ) where
        [(); N_INDUSTRIES * N_INDUSTRIES]:,
        [(); R * R]:,
    {
        let rng = &mut self.rng;
        Sim::get_systematic_shocks::<N_INDUSTRIES>(rng, dyn_data.systematic_shocks_by_cpty);
        Sim::get_idiosynchratic_shocks(rng, dyn_data.idiosynchratic_shocks_by_cpty);
        // Calc factors
        Sim::calculate_thresholds(
            dyn_data.systematic_shocks_by_cpty,
            dyn_data.idiosynchratic_shocks_by_cpty,
            &static_data.r_factor,
            dyn_data.transition_values,
        );
        Sim::assign_pd(
            &static_data.master_scale,
            dyn_data.transition_values,
            dyn_data.assigned_pd,
        );
    }
}

pub struct DynamicSimulationData<'a> {
    systematic_shocks_by_cpty: &'a mut [f32],
    idiosynchratic_shocks_by_cpty: &'a mut [f32],
    transition_values: &'a mut [f32],
    assigned_pd: &'a mut [u8],
}
