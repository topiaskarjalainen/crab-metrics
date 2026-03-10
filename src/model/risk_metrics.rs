use crate::math::{Pcgrnginc, RandomDistribution, RandomGen, STD_NORMAL};

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

pub struct RiskMetrics {
    pub confidence_level: f32,
    pub n_simulations: usize,
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
    fn get_systematic_shocks<const N: usize, T, RNG: RandomGen<T>>(rng: &mut RNG, buf: &mut [f32]);
}

pub struct BaseSimFunctions;

impl SimFunctions for BaseSimFunctions {
    fn get_systematic_shocks<const N: usize>(rng: &mut Pcgrnginc, buf: &mut [f32]) {
        STD_NORMAL.sample_into(rng, buf);
    }
}

impl RiskMetrics {
    pub fn simulate_portfolio_level<
        const N_INDUSTRIES: usize,
        const R: usize,
        T,
        Sim: SimFunctions,
    >(
        &self,
        data: &SimulationData<N_INDUSTRIES, R>,
        rng: &mut impl RandomGen<T>,
    ) -> RiskMetricsResult
    where
        [(); N_INDUSTRIES * N_INDUSTRIES]:,
        [(); R * R]:,
    {
        data.verify_industry_indices(); // Guards against invalid industry indices -> can use unchecked access in the simulation loop
        let mut idiosyncratic_shocks = vec![0f32; data.cpty_ids.len()];
        let systematic_shocks = &mut [0f32; N_INDUSTRIES];
        // Main loop
        for _ in 0..self.n_simulations {
            Sim::get_systematic_shocks::<N_INDUSTRIES, T, _>(rng, systematic_shocks);
        }

        todo!()
    }
}
