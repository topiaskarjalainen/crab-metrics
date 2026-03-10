#![feature(portable_simd)]
#![feature(generic_const_exprs)]

pub mod constants;
pub mod math;
pub mod model;
pub mod simulation;
pub mod util;

const CORRELATION_SIZE: usize = 11;
