use crate::math::{RandomDistribution, RandomGen, STD_NORMAL};

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

impl RiskMetrics {
    pub fn simulate_portfolio_level<const N: usize, const R: usize, T>(
        &self,
        data: &SimulationData<N, R>,
        rng: &mut impl RandomGen<T>,
    ) -> RiskMetricsResult
    where
        [(); N * N]:,
        [(); R * R]:,
    {
        data.verify_industry_indices(); // Guards against invalid industry indices -> can use unchecked access in the simulation loop
        let mut idiosyncratic_shocks = vec![0f32; data.cpty_ids.len()];
        let mut systematic_shocks = vec![0f32; N];
        // Main loop
        for _ in 0..self.n_simulations {
            // STD_NORMAL.sample_into(rng, &mut systematic_shocks);
        }

        todo!()
    }
}
