use std::{
	ffi::{c_char, c_int},
	ptr::null_mut,
};

use crate::{RowMatrix, Vector};

impl<const N: usize> RowMatrix<f64, N, N> {
	pub fn eigen(&self) -> (Vector<f64, N>, RowMatrix<f64, N, N>) {
		let mut copy = self.transpose();

		let n_i32 = N as c_int;
		let mut wr = Vector::<f64, N>::default();
		let mut wi = [0.0; N];
		let vl = null_mut::<f64>(); // not referenced
		let mut vr = RowMatrix::<f64, N, N>::default();
		let mut info: c_int = 0;

		let ljob = b'N' as c_char;
		let rjob = b'V' as c_char;
		let lda = n_i32;
		let ldvl = n_i32;
		let ldvr = n_i32;

		let mut work = vec![0.0; N * 4];
		let lwork = (N * 4) as c_int;

		unsafe {
			lapack_static::dgeev_(
				&ljob,
				&rjob,
				&n_i32,
				copy.as_mut_ptr(),
				&lda,
				wr.as_mut_ptr(),
				wi.as_mut_ptr(),
				vl,
				&ldvl,
				vr.as_mut_ptr(),
				&ldvr,
				work.as_mut_ptr(),
				&lwork,
				&mut info,
			);
		}

		(wr, vr.transpose())
	}

	pub fn inverse(&self) -> Self {
		let mut copy = self.transpose();

		let n_i32 = N as c_int;
		let lda = n_i32;
		let mut ipiv = [0 as c_int; N];
		let mut info: c_int = 0;

		unsafe {
			lapack_static::dgetrf_(
				&n_i32,
				&n_i32,
				copy.as_mut_ptr(),
				&lda,
				ipiv.as_mut_ptr(),
				&mut info,
			)
		}

		assert_eq!(info, 0);

		let mut work = vec![0.0; N * N];
		let lwork = work.len() as c_int;

		unsafe {
			lapack_static::dgetri_(
				&n_i32,
				copy.as_mut_ptr(),
				&n_i32,
				ipiv.as_ptr(),
				work.as_mut_ptr(),
				&lwork,
				&mut info,
			)
		}

		copy.transpose()
	}
}
