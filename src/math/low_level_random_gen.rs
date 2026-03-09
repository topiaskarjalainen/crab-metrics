use std::simd::Simd;
use std::simd::prelude::SimdUint;

pub trait RandomGen<T> {
    type InnerType;

    fn get_next_random(&mut self) -> T;

    fn fill_random(&mut self, buf: &mut [Self::InnerType]);
}


pub struct Pcgrng(u64);

impl Pcgrng {
    pub fn new(seed: u64) -> Self {
        Pcgrng(seed)
    }
}

const MULTIPLIER: u64 = 6364136223846793005;
const INCREMENT: u64 = 1442695040888963407;

const fn multi_splat<const SIZE: usize>() -> Simd<u64, SIZE> {
    Simd::splat(MULTIPLIER)
}

impl RandomGen<u32> for Pcgrng {
    type InnerType = u32;

    fn get_next_random(&mut self) -> u32 {
        let old_state = self.0;
        self.0 = old_state.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        let xorshifted = (((old_state >> 18u32) ^ old_state) >> 27u32) as u32;
        let rot = (old_state >> 59u32) as u32;
        (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))
    }

    fn fill_random(&mut self, buf: &mut [Self::InnerType]) {
        for x in buf.iter_mut() {
            *x = self.get_next_random();
        }
    }
}


impl RandomGen<f32> for Pcgrng {
    type InnerType = f32;

    fn get_next_random(&mut self) -> f32 {
        let a = RandomGen::<u32>::get_next_random(self);
        let masked = a & ((1 << 24) - 1);
        masked as f32 * 2f32.powi(-24)
    }

    fn fill_random(&mut self, buf: &mut [Self::InnerType]) {
        for x in buf.iter_mut() {
            *x = self.get_next_random();
        }
    }
}

pub struct Pcgrnginc(u64, u64);

impl Pcgrnginc {
    pub fn new(seed: u64, inc: u64) -> Self {
        Pcgrnginc(seed, inc)
    }
}

impl RandomGen<u32> for Pcgrnginc {
    type InnerType = u32;

    fn get_next_random(&mut self) -> u32 {
        let old_state = self.0;
        self.0 = old_state.wrapping_mul(MULTIPLIER).wrapping_add(self.1);
        let xorshifted = (((old_state >> 18u32) ^ old_state) >> 27u32) as u32;
        let rot = (old_state >> 59u32) as u32;
        (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))
    }

    fn fill_random(&mut self, buf: &mut [Self::InnerType]) {
        for x in buf.iter_mut() {
            *x = self.get_next_random();
        }
    }
}

pub struct SimdPcgrng<const SIZE: usize>(Simd<u64, SIZE>);

impl<const SIZE: usize> SimdPcgrng<SIZE> {
    const MULTI_SPLAT: Simd<u64, SIZE> = multi_splat::<SIZE>();

    pub fn new(seed: Simd<u64, SIZE>) -> Self {
        SimdPcgrng(seed)
    }
}

#[inline(always)]
fn _rotate_right<const SIZE: usize>(x: Simd<u32, SIZE>, rot: Simd<u32, SIZE>) -> Simd<u32, SIZE> {
    let a = (x >> rot); //| (x << ((-(rot as i32)) & 31))
    let b = rot.wrapping_neg() & Simd::splat(31);
    let c = x << b;
    a | c
}

impl<const SIZE: usize> RandomGen<Simd<u32, SIZE>> for SimdPcgrng<SIZE> {
    type InnerType = u32;

    fn get_next_random(&mut self) -> Simd<u32, SIZE> {
        let old_state = self.0;
        self.0 = old_state * Self::MULTI_SPLAT + Simd::splat(INCREMENT | 1);
        let xorshifted = ((old_state >> Simd::splat(18)) ^ old_state) >> Simd::splat(27);
        let rot = old_state >> Simd::splat(59);
        _rotate_right(xorshifted.cast(), rot.cast())
    }

    fn fill_random(&mut self, buf: &mut [Self::InnerType]) {
        let (val, rem) = buf.as_chunks_mut::<SIZE>();
        for chunk in val {
            let randoms: Simd<u32, SIZE> = self.get_next_random();
            chunk.copy_from_slice(&randoms.to_array());
        }
        // Remainder handling if buf length is not a multiple of SIZE
        let last_values: [u32; SIZE] = RandomGen::<Simd<u32, SIZE>>::get_next_random(self).to_array();
        for (i, rem) in rem.iter_mut().enumerate() {
                *rem = last_values[i];
        }
    }
}


impl<const SIZE: usize> RandomGen<Simd<f32, SIZE>> for SimdPcgrng<SIZE> {
    type InnerType = f32;

    fn get_next_random(&mut self) -> Simd<f32, SIZE> {
        let a = RandomGen::<Simd<u32, SIZE>>::get_next_random(self);
        let masked = a & Simd::splat((1 << 24) - 1);
        masked.cast::<f32>() * Simd::splat(2f32.powi(-24))
    }

    fn fill_random(&mut self, buf: &mut [Self::InnerType]) {
        let (val, rem) = buf.as_chunks_mut::<SIZE>();
        for chunk in val {
            let randoms: Simd<f32, SIZE> = self.get_next_random();
            chunk.copy_from_slice(&randoms.to_array());
        }
        // Remainder handling if buf length is not a multiple of SIZE
        let last_values: [f32; SIZE] = RandomGen::<Simd<f32, SIZE>>::get_next_random(self).to_array();
        for (i, rem) in rem.iter_mut().enumerate() {
                *rem = last_values[i];
        }
    }

}
