use std::simd::cmp::SimdPartialOrd;
use std::simd::{Mask, Simd};

pub trait AssignFromArray {
    /// Assign for all in elems the index of the first larger or equal value in look_up,
    /// or N if no such value exists. Output is written to out. Assume N <= 32 and elems.len() == out.len()
    /// and look_up is sorted in ascending order.
    fn assign_from_array<const N: usize>(look_up: &[f32; N], elems: &[f32], out: &mut [u8]);
}

pub struct NaiveAssign;

impl AssignFromArray for NaiveAssign {
    fn assign_from_array<const N: usize>(look_up: &[f32; N], elems: &[f32], out: &mut [u8]) {
        for (i, &elem) in elems.iter().enumerate() {
            let mut assigned = N as u8; // Default to N if no larger or equal value is found
            for (j, &val) in look_up.iter().enumerate() {
                if elem <= val {
                    assigned = j as u8;
                    break;
                }
            }
            out[i] = assigned;
        }
    }
}

pub struct StdAssign;

impl AssignFromArray for StdAssign {
    fn assign_from_array<const N: usize>(look_up: &[f32; N], elems: &[f32], out: &mut [u8]) {
        for (i, &elem) in elems.iter().enumerate() {
            let assigned = look_up.iter().position(|&val| elem <= val).unwrap_or(N) as u8; // Default to N if no larger or equal value is found
            out[i] = assigned;
        }
    }
}

pub struct SimdAssign<const LANES: usize>;

#[inline(always)]
fn _impl_assign_8<const N: usize>(look_up: &[f32; N], elems: &[f32], out: &mut [u8]) {
    todo!()
}

#[inline(always)]
fn _impl_assign_16<const N: usize>(look_up: &[f32; N], elems: &[f32], out: &mut [u8]) {
    // We'll assume 32 >= N > 16, -> Two SIMD registers are enough to hold the look_up values, and we can process 16 elements at a time.
    type Vx = Simd<f32, 16>;
    type Mx = Mask<i32, 16>;
    let v_lookup1 = Vx::from_slice(&look_up[0..16]);
    let v_lookup2 = Vx::load_or(&look_up[16..], Simd::splat(f32::MAX));
    for (&input, output) in elems.iter().zip(out.iter_mut()) {
        let v_input = Vx::splat(input);
        let mask1 = v_input.simd_le(v_lookup1);
        let mask2 = v_input.simd_le(v_lookup2);

        let bm1 = mask1.to_bitmask() as u32; // 16 bits for the first 16 values
        let bm2 = mask2.to_bitmask() as u32; // 16 bits

        let idx1 = bm1.trailing_zeros(); // Position of the first 1 bit in mask1
        let idx2 = bm2.trailing_zeros(); // Position of the first 1 bit in mask2

        let assigned = if idx1 < 16 {
            idx1 as u8
        } else if idx2 < 16 {
            (16 + idx2) as u8
        } else {
            (N - 1) as u8 // Default to N if no larger or equal value is found
        };

        *output = assigned;
    }
}

impl<const LANES: usize> AssignFromArray for SimdAssign<LANES> {
    fn assign_from_array<const N: usize>(look_up: &[f32; N], elems: &[f32], out: &mut [u8]) {
        if LANES == 8 {
            _impl_assign_8(look_up, elems, out);
        } else if LANES == 16 {
            _impl_assign_16(look_up, elems, out);
        } else {
            panic!("Unsupported LANES value for SimdAssign");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign() {
        let look_up = [
            0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ];
        let elems = [0.5, 0.0, 0.3, 1.2, 5.0];
        let mut out = [0u8; 5];
        NaiveAssign::assign_from_array(&look_up, &elems, &mut out);
        dbg!(&out);
    }

    #[test]
    fn assign_simd() {
        let look_up = [
            0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, //
            2.3, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ];
        let elems = [0.5, 0.0, 0.3, 1.2, 5.0];
        let mut out = [0u8; 5];
        let mut out_std = [0u8; 5];
        StdAssign::assign_from_array(&look_up, &elems, &mut out_std);
        SimdAssign::<16>::assign_from_array(&look_up, &elems, &mut out);
        assert_eq!(out, out_std);
    }
}
