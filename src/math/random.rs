use std::simd::{Simd, StdFloat};
use crate::math::RandomGen;

pub trait RandomDistribution<T> {
    fn sample<RNG: RandomGen<T>>(&self, rng: &mut RNG) -> T;
    fn sample_into<RNG: RandomGen<T>>(&self, rng: &mut RNG, buf: &mut [RNG::InnerType]);
}


#[derive(Debug, Clone, Copy)]
pub struct RandomNormal(f32, f32);

pub const STD_NORMAL: RandomNormal = RandomNormal(0.0, 1.0);

impl RandomNormal {
    pub fn new(mean: f32, std_dev: f32) -> Self {
        RandomNormal(mean, std_dev)
    }
}

impl RandomDistribution<f32> for RandomNormal {
    fn sample<RNG: RandomGen<f32>>(&self, rng: &mut RNG) -> f32 {
        // Box-Muller transform
        let u1 = rng.get_next_random();
        let u2 = rng.get_next_random();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        self.0 + z0 * self.1
    }

    fn sample_into<RNG: RandomGen<f32>>(&self, rng: &mut RNG, buf: &mut [RNG::InnerType]) {
        for x in buf.iter_mut() {
            // *x = self.sample(rng);
            todo!()
        }
    }
}


pub struct SimdRandomNormal(f32, f32);

impl SimdRandomNormal {
    const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

    pub fn new(mean: f32, std_dev: f32) -> Self {
        SimdRandomNormal(mean, std_dev)
    }
}

impl<const LANES: usize> RandomDistribution<Simd<f32, LANES>> for SimdRandomNormal {
    fn sample<RNG: RandomGen<Simd<f32, LANES>>>(&self, rng: &mut RNG) -> Simd<f32, LANES> {
        // // Box-Muller transform for SIMD
        let u1 = RandomGen::<Simd<f32, LANES>>::get_next_random(rng);
        let u2 = RandomGen::<Simd<f32, LANES>>::get_next_random(rng);
        let z0 = (Simd::splat(-2.0) * u1.ln()).sqrt() * (Simd::splat(Self::TWO_PI) * u2).cos();
        Simd::splat(self.0) + z0 * Simd::splat(self.1)
    }

    fn sample_into<RNG: RandomGen<Simd<f32, LANES>>>(&self, rng: &mut RNG, buf: &mut [RNG::InnerType]) {
        let (val, rem) = buf.as_chunks_mut::<LANES>();
        for chunk in val.iter_mut() {
            let sample = self.sample(rng);
            // chunk.copy_from_slice(&sample.to_array());
        }
        //rem`
        let last_sample = self.sample(rng);
        for (i, x) in rem.iter_mut().enumerate() {
            // *x = last_sample[i];
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::math::{Pcgrng, SimdPcgrng};
    use super::*;

    #[test]
    fn float_normal() {
        let mut rng = Pcgrng::new(0x12345678);
        let dist = RandomNormal::new(0.0, 1.0);
        for _ in 0..10 {
            let sample = dist.sample(&mut rng);
            println!("Sample: {}", sample);
        }
    }

    #[test]
    fn simd_normal() {
        const LANES: usize = 8;
        let seed = [
            0x12345678, 0x9abcdef0, 0xdeadbeef, 0xfeedface,
            0xabcdef01, 0x23456789, 0x34567890, 0x45678901,
        ];
        let mut rng = SimdPcgrng::new(seed.into());
        let dist = SimdRandomNormal::new(0.0, 1.0);
        for _ in 0..10 {
            let sample: Simd<f32, LANES> = dist.sample(&mut rng);
            println!("Sample: {:?}", sample);
        }
    }

}


