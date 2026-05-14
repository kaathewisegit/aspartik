#![expect(clippy::needless_range_loop)]

use crate::matrix::{Matrix, SquareMatrix};

#[derive(Debug)]
pub struct EigenDecomposition<M, const N: usize>
where
	M: Matrix<f64, N, N>,
{
	pub eigenvalues: [f64; N],
	pub eigenvalues_img: [f64; N],
	pub eigenvectors: M,
}

pub fn eigen<M, const N: usize>(matrix: &M) -> EigenDecomposition<M, N>
where
	M: SquareMatrix<f64, N>,
{
	let (mut h, mut z): (M, M) = hessenberg_reduce_with_z(matrix);

	francis_qr(&mut h, &mut z);

	let (eigenvalues_real, eigenvalues_imag) = tung_tung_tung_schur(&h);

	let mut evecs_schur = M::zeros();
	let mut i = 0;
	while i < N {
		if eigenvalues_imag[i].abs() > 1e-10 {
			let (yr, yi) = schur_complex_eigenvec(
				&h,
				i,
				eigenvalues_real[i],
				eigenvalues_imag[i],
			);
			for r in 0..N {
				*evecs_schur.at_mut(r, i) = yr[r];
				*evecs_schur.at_mut(r, i + 1) = yi[r];
			}
			i += 2;
		} else {
			let y = schur_real_eigenvec(&h, i);
			for r in 0..N {
				*evecs_schur.at_mut(r, i) = y[r];
			}
			i += 1;
		}
	}

	let eigenvectors: M = z.mul(&evecs_schur);

	EigenDecomposition {
		eigenvalues: eigenvalues_real,
		eigenvalues_img: eigenvalues_imag,
		eigenvectors,
	}
}

fn schur_real_eigenvec<M, const N: usize>(t: &M, p: usize) -> [f64; N]
where
	M: Matrix<f64, N, N>,
{
	let lambda = *t.at(p, p);
	let mut y = [0.0f64; N];
	y[p] = 1.0;
	for i in (0..p).rev() {
		let mut acc = 0.0f64;
		for j in (i + 1)..=p {
			acc -= *t.at(i, j) * y[j];
		}
		let diag = *t.at(i, i) - lambda;
		y[i] = if diag.abs() > 1e-14 { acc / diag } else { 0.0 };
	}

	let norm =
		y.iter().map(|x| x * x)
			.sum::<f64>()
			.sqrt()
			.max(f64::EPSILON);
	y.iter_mut().for_each(|x| *x /= norm);
	y
}

fn schur_complex_eigenvec<M, const N: usize>(
	t: &M,
	p: usize,
	re: f64,
	im: f64,
) -> ([f64; N], [f64; N])
where
	M: Matrix<f64, N, N>,
{
	let mut yr = [0.0f64; N];
	let mut yi = [0.0f64; N];
	yr[p] = 1.0;
	yi[p + 1] = 1.0;

	for i in (0..p).rev() {
		let mut accr = 0.0f64;
		let mut acci = 0.0f64;
		for j in (i + 1)..N {
			let tij = *t.at(i, j);
			accr -= tij * yr[j];
			acci -= tij * yi[j];
		}
		let dre = *t.at(i, i) - re;
		let det = dre * dre + im * im;
		if det > 1e-28 {
			yr[i] = (dre * accr + im * acci) / det;
			yi[i] = (dre * acci - im * accr) / det;
		}
	}

	let norm = (yr.iter().chain(yi.iter()).map(|x| x * x).sum::<f64>())
		.sqrt()
		.max(f64::EPSILON);
	yr.iter_mut().for_each(|x| *x /= norm);
	yi.iter_mut().for_each(|x| *x /= norm);
	(yr, yi)
}

fn hessenberg_reduce_with_z<M, const N: usize>(a: &M) -> (M, M)
where
	M: SquareMatrix<f64, N> + Matrix<f64, N, N>,
{
	let mut h = a.clone();
	let mut z = M::identity();

	for k in 0..N.saturating_sub(2) {
		let mut norm_sq = 0.0f64;
		for i in (k + 1)..N {
			norm_sq += *h.at(i, k) * *h.at(i, k);
		}
		let norm = norm_sq.sqrt();
		if norm < f64::EPSILON {
			continue;
		}

		let mut v = [0.0f64; N];
		for i in (k + 1)..N {
			v[i] = *h.at(i, k);
		}
		let sign = if v[k + 1] >= 0.0 { 1.0 } else { -1.0 };
		v[k + 1] += sign * norm;

		let vv: f64 = ((k + 1)..N).map(|i| v[i] * v[i]).sum();
		if vv < f64::EPSILON {
			continue;
		}
		let two_over_vv = 2.0 / vv;

		for j in k..N {
			let mut dot = 0.0f64;
			for i in (k + 1)..N {
				dot += v[i] * *h.at(i, j);
			}
			let f = two_over_vv * dot;
			for i in (k + 1)..N {
				*h.at_mut(i, j) -= f * v[i];
			}
		}

		for i in 0..N {
			let mut dot = 0.0f64;
			for j in (k + 1)..N {
				dot += *h.at(i, j) * v[j];
			}
			let f = two_over_vv * dot;
			for j in (k + 1)..N {
				*h.at_mut(i, j) -= f * v[j];
			}
		}

		for i in 0..N {
			let mut dot = 0.0f64;
			for j in (k + 1)..N {
				dot += *z.at(i, j) * v[j];
			}
			let f = two_over_vv * dot;
			for j in (k + 1)..N {
				*z.at_mut(i, j) -= f * v[j];
			}
		}
	}
	(h, z)
}

fn francis_qr<M, const N: usize>(h: &mut M, z: &mut M)
where
	M: SquareMatrix<f64, N> + Matrix<f64, N, N>,
{
	let mut zt: M = z.transpose();
	let mut active_end = N;
	let mut iter = 0;
	let max_iter = 100 * N;

	while active_end > 1 && iter < max_iter {
		let active_start = find_deflation_point(h, active_end);

		if active_end - active_start <= 2 {
			if active_end - active_start == 2 {
				let p = active_start;
				let a = *h.at(p, p);
				let b = *h.at(p, p + 1);
				let c = *h.at(p + 1, p);
				let d = *h.at(p + 1, p + 1);
				if c.abs() > f64::EPSILON * (a.abs() + d.abs())
				{
					let tr = a + d;
					let det = a * d - b * c;
					let disc = tr * tr - 4.0 * det;
					if disc >= 0.0 {
						standardize_2x2_real(
							h, &mut zt, p,
						);
					}
				}
			}
			active_end = active_start;
			continue;
		}

		let (sigma_sum, sigma_prod) = if iter % 10 == 9 {
			let s1 = h.at(active_end - 1, active_end - 2).abs();
			let s2 = if active_end >= 3 {
				h.at(active_end - 2, active_end - 3).abs()
			} else {
				0.0
			};
			let t = s1 + s2;
			(1.5 * t, 0.8125 * t * t)
		} else {
			wilkinson_shift(h, active_end)
		};

		francis_double_step(
			h,
			&mut zt,
			active_start,
			active_end,
			sigma_sum,
			sigma_prod,
		);
		iter += 1;
	}
	*z = zt.transpose();
}

fn standardize_2x2_real<M, const N: usize>(h: &mut M, zt: &mut M, p: usize)
where
	M: Matrix<f64, N, N>,
{
	let a = *h.at(p, p);
	let b = *h.at(p, p + 1);
	let c = *h.at(p + 1, p);
	let d = *h.at(p + 1, p + 1);
	let tr = a + d;
	let det = a * d - b * c;
	let disc = tr * tr - 4.0 * det;
	let sq = disc.max(0.0).sqrt();
	let lam1 = (tr + sq) * 0.5;
	let (vx, vy) = if b.hypot(lam1 - a) >= (lam1 - d).hypot(c) {
		(b, lam1 - a)
	} else {
		(lam1 - d, c)
	};
	let norm = vx.hypot(vy);
	if norm < f64::EPSILON {
		return;
	}
	let cs = vx / norm;
	let sn = vy / norm;

	for j in 0..N {
		let t0 = cs * *h.at(p, j) + sn * *h.at(p + 1, j);
		let t1 = -sn * *h.at(p, j) + cs * *h.at(p + 1, j);
		*h.at_mut(p, j) = t0;
		*h.at_mut(p + 1, j) = t1;
	}
	for i in 0..N {
		let t0 = cs * *h.at(i, p) + sn * *h.at(i, p + 1);
		let t1 = -sn * *h.at(i, p) + cs * *h.at(i, p + 1);
		*h.at_mut(i, p) = t0;
		*h.at_mut(i, p + 1) = t1;
	}
	*h.at_mut(p + 1, p) = 0.0;
	for j in 0..N {
		let t0 = cs * *zt.at(p, j) + sn * *zt.at(p + 1, j);
		let t1 = -sn * *zt.at(p, j) + cs * *zt.at(p + 1, j);
		*zt.at_mut(p, j) = t0;
		*zt.at_mut(p + 1, j) = t1;
	}
}

fn find_deflation_point<M, const N: usize>(
	h: &mut M,
	active_end: usize,
) -> usize
where
	M: Matrix<f64, N, N>,
{
	for i in (0..active_end.saturating_sub(1)).rev() {
		let sub = h.at(i + 1, i).abs();
		let scale = h.at(i, i).abs() + h.at(i + 1, i + 1).abs();
		if sub <= f64::EPSILON * scale {
			*h.at_mut(i + 1, i) = 0.0;
			return i + 1;
		}
	}
	0
}

fn wilkinson_shift<M, const N: usize>(h: &M, end: usize) -> (f64, f64)
where
	M: Matrix<f64, N, N>,
{
	let a = *h.at(end - 2, end - 2);
	let d = *h.at(end - 1, end - 1);
	(
		a + d,
		a * d - *h.at(end - 2, end - 1) * *h.at(end - 1, end - 2),
	)
}

fn francis_double_step<M, const N: usize>(
	h: &mut M,
	zt: &mut M,
	start: usize,
	end: usize,
	shift_sum: f64,
	shift_prod: f64,
) where
	M: Matrix<f64, N, N>,
{
	let (s, e) = (start, end);
	let sub = *h.at(s + 1, s);
	let p0 = *h.at(s, s) * *h.at(s, s) + *h.at(s, s + 1) * sub
		- shift_sum * *h.at(s, s)
		+ shift_prod;
	let p1 = sub * (*h.at(s, s) + *h.at(s + 1, s + 1) - shift_sum);
	let p2 = *h.at(s + 2, s + 1) * sub;
	let mut x = [p0, p1, p2];

	for k in 0..(e - s - 2) {
		let r = s + k;
		let mut v = x;
		let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
		if norm < f64::EPSILON {
			if k + 1 < e - s - 2 {
				x = [
					*h.at(r + 1, r),
					*h.at(r + 2, r),
					*h.at(r + 3, r),
				];
			}
			continue;
		}
		let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
		v[0] += sign * norm;
		let two_over_vv =
			2.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

		let col_start = if k == 0 { r } else { r - 1 };
		for j in col_start..e {
			let dot = v[0] * *h.at(r, j)
				+ v[1] * *h.at(r + 1, j) + v[2] * *h
				.at(r + 2, j);
			let f = two_over_vv * dot;
			*h.at_mut(r, j) -= f * v[0];
			*h.at_mut(r + 1, j) -= f * v[1];
			*h.at_mut(r + 2, j) -= f * v[2];
		}
		for i in 0..(r + 4).min(e) {
			let dot = v[0] * *h.at(i, r)
				+ v[1] * *h.at(i, r + 1) + v[2] * *h
				.at(i, r + 2);
			let f = two_over_vv * dot;
			*h.at_mut(i, r) -= f * v[0];
			*h.at_mut(i, r + 1) -= f * v[1];
			*h.at_mut(i, r + 2) -= f * v[2];
		}
		for j in 0..N {
			let dot = v[0] * *zt.at(r, j)
				+ v[1] * *zt.at(r + 1, j) + v[2] * *zt
				.at(r + 2, j);
			let f = two_over_vv * dot;
			*zt.at_mut(r, j) -= f * v[0];
			*zt.at_mut(r + 1, j) -= f * v[1];
			*zt.at_mut(r + 2, j) -= f * v[2];
		}
		if k + 1 < e - s - 2 {
			x = [*h.at(r + 1, r), *h.at(r + 2, r), *h.at(r + 3, r)];
		}
	}

	let (rt, rb, cf) = (e - 2, e - 1, e - 3);
	let (x0, v1) = (*h.at(rt, cf), *h.at(rb, cf));
	let n2 = (x0 * x0 + v1 * v1).sqrt();
	if n2 > f64::EPSILON {
		let v0 = x0 + (if x0 >= 0.0 { 1.0 } else { -1.0 }) * n2;
		let two_over_vv = 2.0 / (v0 * v0 + v1 * v1);
		for j in cf..N {
			let dot = v0 * *h.at(rt, j) + v1 * *h.at(rb, j);
			let f = two_over_vv * dot;
			*h.at_mut(rt, j) -= f * v0;
			*h.at_mut(rb, j) -= f * v1;
		}
		for i in 0..e {
			let dot = v0 * *h.at(i, rt) + v1 * *h.at(i, rb);
			let f = two_over_vv * dot;
			*h.at_mut(i, rt) -= f * v0;
			*h.at_mut(i, rb) -= f * v1;
		}
		for j in 0..N {
			let dot = v0 * *zt.at(rt, j) + v1 * *zt.at(rb, j);
			let f = two_over_vv * dot;
			*zt.at_mut(rt, j) -= f * v0;
			*zt.at_mut(rb, j) -= f * v1;
		}
	}
}

fn tung_tung_tung_schur<M, const N: usize>(t: &M) -> ([f64; N], [f64; N])
where
	M: Matrix<f64, N, N>,
{
	let (mut re, mut im) = ([0.0; N], [0.0; N]);
	let mut i = 0;
	while i < N {
		let is_2x2 = i + 1 < N
			&& t.at(i + 1, i).abs()
				> f64::EPSILON
					* (t.at(i, i).abs()
						+ t.at(i + 1, i + 1).abs());
		if is_2x2 {
			let (a, d) = (*t.at(i, i), *t.at(i + 1, i + 1));
			let det = a * d - *t.at(i, i + 1) * *t.at(i + 1, i);
			let tr = a + d;
			let disc = tr * tr - 4.0 * det;
			if disc >= 0.0 {
				re[i] = *t.at(i, i);
				re[i + 1] = *t.at(i + 1, i + 1);
			} else {
				re[i] = tr * 0.5;
				re[i + 1] = tr * 0.5;
				im[i] = (-disc).sqrt() * 0.5;
				im[i + 1] = -im[i];
			}
			i += 2;
		} else {
			re[i] = *t.at(i, i);
			i += 1;
		}
	}
	(re, im)
}
