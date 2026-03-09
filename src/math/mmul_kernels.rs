use std::simd::num::SimdFloat;
use std::simd::{Mask, Select, Simd};

pub trait GEMVKernel {
    fn gemv(a: &[f32], x: &[f32], y: &mut [f32]);
}


pub struct NaiveGEMV<const ROW_MAJOR: bool>;

impl<const ROW_MAJOR: bool> GEMVKernel for NaiveGEMV<ROW_MAJOR> {
    fn gemv(a: &[f32], x: &[f32], y: &mut [f32]) {
        let m = y.len();
        let n = x.len();
        for i in 0..m {
            let mut sum = 0f32;
            for j in 0..n {
                let idx = if ROW_MAJOR {
                    i * n + j
                } else {
                    j * m + i
                };
                sum += a[idx] * x[j];
            }
            y[i] = sum;
        }
    }
}

pub struct NaiveGEMVUnsafe<const ROW_MAJOR: bool>;

impl<const ROW_MAJOR: bool> GEMVKernel for NaiveGEMVUnsafe<ROW_MAJOR> {
    fn gemv(a: &[f32], x: &[f32], y: &mut [f32]) {
        let m = y.len();
        let n = x.len();
        for i in 0..m {
            let mut sum = 0f32;
            for j in 0..n {
                let idx = if ROW_MAJOR {
                    i * n + j
                } else {
                    j * m + i
                };
                unsafe { sum += a.get_unchecked(idx) * x.get_unchecked(j); }
                // sum += a[idx] * x[j];
            }
            y[i] = sum;
        }
    }
}



/// Assume row-major SIZE x SIZE matrix where only the lower triangle (including diagonal) is stored
pub struct ManualStaticLowerTriangularGEMV<const SIZE: usize, const LANES: usize>;

impl<const SIZE: usize, const LANES: usize> ManualStaticLowerTriangularGEMV<SIZE, LANES> {
    #[inline(always)]
    fn impl_gemv(a: &[f32], x: &[f32], y: &mut [f32]) {
        // Iter by row
        // Tail should be empty since we assmue SIZE x SIZE matrix and SIZE < LANES
        let (head, _) = a.as_chunks::<SIZE>();
        let x_simd = Simd::<f32, LANES>::load_or_default(x);
        let mut sum_simd = Simd::<f32, LANES>::splat(0.0);
        let mut result_mask: Mask<i32, LANES> = Mask::splat(false);
        for (_, slice) in head.iter().enumerate() {
            let a_row = Simd::<f32, LANES>::load_or_default(slice);
            let sum = (a_row * x_simd).reduce_sum();
            sum_simd = result_mask.select(Simd::<f32, LANES>::splat(sum), sum_simd);
            result_mask = result_mask.rotate_elements_left::<1>();
        }
        y.copy_from_slice(&sum_simd.to_array()[..SIZE]);
    }
}

impl<const SIZE: usize, const LANES: usize> GEMVKernel for ManualStaticLowerTriangularGEMV<SIZE, LANES> {
    fn gemv(a: &[f32], x: &[f32], y: &mut [f32]) {
        // const _: () = {
        //     assert!(SIZE > 0, "SIZE must be greater than 0");
        //     assert!(LANES > 0, "LANES must be greater than 0");
        //     assert!(SIZE < LANES, "SIZE must be smaller than LANES");
        // };
        assert!(SIZE < LANES, "SIZE must be smaller than LANES");
        assert_eq!(a.len(), SIZE * SIZE);
        assert_eq!(x.len(), SIZE);
        assert_eq!(y.len(), SIZE);
        Self::impl_gemv(a, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive() {
        const M: usize = 2;
        const N: usize = 3;
        let a: [f32; M * N] = [
            1., 2., 3., // Row 0
            4., 5., 6., // Row 1
        ];
        let x: [f32; N] = [1., 1., 1.];
        let mut y: [f32; M] = [0.; M];
        NaiveGEMV::<true>::gemv(&a, &x, &mut y);
        assert_eq!(y, [6., 15.]);
        dbg!(&y);
    }
}
