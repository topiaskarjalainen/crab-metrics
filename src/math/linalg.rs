

pub fn cholesky_decomposition<const DIM: usize>(mat: &[[f32; DIM]; DIM]) -> [[f32; DIM]; DIM] {
    let mut l = [[0.0; DIM]; DIM];

    for i in 0..DIM {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = (mat[i][i] - sum).sqrt();
            } else {
                l[i][j] = (mat[i][j] - sum) / l[j][j];
            }
        }
    }

    l
}


pub fn print_matrix<const ROW: usize, const COL: usize>(mat: &[[f32; COL]; ROW]) {
    for i in 0..ROW {
        for j in 0..COL {
            print!("{:8.4} ", mat[i][j]);
        }
        println!();
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constcholesky() {
        let mat = [
            [4., 12., -16.],
            [12., 37., -43.],
            [-16., -43., 98.]
        ];

        let expected = [
            [2., 0., 0.],
            [6., 1., 0.],
            [-8., 5., 3.]
        ];
        let decomp = cholesky_decomposition::<3>(&mat);
        assert_eq!(decomp, expected);
        print_matrix::<3, 3>(&expected);
        print_matrix::<3, 3>(&decomp);
    }
}
