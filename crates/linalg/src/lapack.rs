use std::{
	ffi::{c_char, c_int},
	ptr::null_mut,
};

use crate::{
	Vector,
	matrix::{Matrix, transpose},
};

pub fn eigen<const N: usize>(
	m: &impl Matrix<f64, N, N>,
) -> ([f64; N], [[f64; N]; N]) {
	let mut copy: [[f64; N]; N] = transpose(m);

	let n_i32 = N as c_int;
	let mut wr = <[f64; N]>::from_element(0.0);
	let mut wi = [0.0; N];
	let vl = null_mut::<f64>(); // not referenced
	let mut vr: [[f64; N]; N] = Matrix::zeros();
	let mut info: c_int = 0;

	let ljob = b'N' as c_char;
	let rjob = b'V' as c_char;
	let lda = n_i32;
	let ldvl = n_i32;
	let ldvr = n_i32;

	let mut work = [[0.0; 4]; N]; // has the length of N * 4
	let lwork = (N * 4) as c_int;

	// SAFETY:
	// - `ljob` and `rjob` are valid characters
	// - `copy` is a contiguous memory slice the size of `N^2`
	// - `wr` and `wi` are contiguous memory slices the size of `N`
	// - `vl` won't be referenced because `ljob` is `N`
	// - `vr` has the same size as `copy` and `lvdl` is `N`
	// - The size of `work` is the same as `lwork`, `N * 4`
	//
	// https://netlib.org/lapack/explore-html/d4/d68/group__geev_ga7d8afe93d23c5862e238626905ee145e.html
	unsafe {
		lapack_static::dgeev_(
			&ljob,
			&rjob,
			&n_i32,
			copy.as_mut_ptr() as *mut f64,
			&lda,
			wr.as_mut_ptr(),
			wi.as_mut_ptr(),
			vl,
			&ldvl,
			vr.as_mut_ptr() as *mut f64,
			&ldvr,
			work.as_mut_ptr() as *mut f64,
			&lwork,
			&mut info,
		);
	}

	(wr, transpose(&vr))
}

pub fn inverse<const N: usize>(m: &impl Matrix<f64, N, N>) -> [[f64; N]; N] {
	let mut copy: [[f64; N]; N] = transpose(m);

	let n_i32 = N as c_int;
	let lda = n_i32;
	let mut ipiv = [0 as c_int; N];
	let mut info: c_int = 0;

	// SAFETY:
	// - `n_i32` is `N`
	// - `copy` has the size of `N^2`
	// - `ipiv` has the length of `N`, `lda` is also `N`.
	//
	// https://netlib.org/lapack/explore-html/db/d04/group__getrf_gaea332d65e208d833716b405ea2a1ab69.html
	unsafe {
		lapack_static::dgetrf_(
			&n_i32,
			&n_i32,
			copy.as_mut_ptr() as *mut f64,
			&lda,
			ipiv.as_mut_ptr(),
			&mut info,
		)
	}
	assert_eq!(info, 0);

	let mut work = [[0.0; N]; N];
	let lwork = work.len() as c_int;

	// SAFETY:
	// - `copy` has the size of `N^2`
	// - the size of `work` is `N^2`, `lwork` is `N^2`
	// - `ipiv` comes from `dgetrf`
	//
	// https://netlib.org/lapack/explore-html/da/d28/group__getri_ga8b6904853957cfb37eba1e72a3f372d2.html
	unsafe {
		lapack_static::dgetri_(
			&n_i32,
			copy.as_mut_ptr() as *mut f64,
			&n_i32,
			ipiv.as_ptr(),
			work.as_mut_ptr() as *mut f64,
			&lwork,
			&mut info,
		)
	}
	assert_eq!(info, 0);

	transpose(&copy)
}
