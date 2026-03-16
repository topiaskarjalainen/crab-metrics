#![feature(portable_simd)]
#![feature(generic_const_exprs)]

use crab_metrics::model::risk_metrics as rm;

pub fn main() {
    divan::main();
}

const N_IND: usize = 11;
const N_RATINGS: usize = 18;

fn _get_data(n: usize) -> rm::StaticSimulationData<'static, N_IND, N_RATINGS>
where
    [(); N_IND * N_IND]:,
    [(); N_RATINGS * N_RATINGS]:,
{
    todo!()
}
