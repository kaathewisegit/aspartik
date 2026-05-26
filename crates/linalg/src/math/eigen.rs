use super::mul;
use crate::{MatrixMut, MatrixRef};

pub fn eigen<'a, I, V>(
	matrix: I,
	eigenvalues: &mut [f64],
	eigenvalues_img: &mut [f64],
	eigenvectors: V,
) where
	I: Into<MatrixRef<'a, f64>>,
	V: Into<MatrixMut<'a, f64>>,
{
	eigen_inner(
		matrix.into(),
		eigenvalues,
		eigenvalues_img,
		eigenvectors.into(),
	)
}

fn eigen_inner(
	matrix: MatrixRef<f64>,

	eigenvalues: &mut [f64],
	eigenvalues_img: &mut [f64],
	eigenvectors: MatrixMut<f64>,
) {
	assert!(matrix.is_square());
	let n = matrix.num_rows();

	let (mut h_box, mut z_box) = hessenberg_reduce_with_z(matrix);
	let mut h = MatrixMut::from_slice(&mut h_box, n, n);
	let mut z = MatrixMut::from_slice(&mut z_box, n, n);

	francis_qr(h.reborrow(), z.reborrow());
	let h = *h;

	tung_tung_tung_schur(h, eigenvalues, eigenvalues_img);

	let mut evecs_schur = vec![0.0; n * n];
	let mut evecs_schur = MatrixMut::from_slice(&mut evecs_schur, n, n);
	let mut i = 0;
	while i < n {
		if eigenvalues_img[i].abs() > 1e-10 {
			let (yr, yi) = schur_complex_eigenvec(
				h,
				i,
				eigenvalues[i],
				eigenvalues_img[i],
			);
			for r in 0..n {
				evecs_schur[(r, i)] = yr[r];
				evecs_schur[(r, i + 1)] = yi[r];
			}
			i += 2;
		} else {
			let y = schur_real_eigenvec(h, i);
			for r in 0..n {
				evecs_schur[(r, i)] = y[r];
			}
			i += 1;
		}
	}

	mul(*z, *evecs_schur, eigenvectors);
}

fn schur_real_eigenvec(t: MatrixRef<f64>, p: usize) -> Vec<f64> {
	let n = t.num_rows();
	let lambda = t[(p, p)];
	let mut y = vec![0.0f64; n];
	y[p] = 1.0;

	for i in (0..p).rev() {
		let mut acc = 0.0f64;
		for j in (i + 1)..=p {
			acc -= t[(i, j)] * y[j];
		}
		let diag = t[(i, i)] - lambda;
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

fn schur_complex_eigenvec(
	t: MatrixRef<f64>,
	p: usize,
	re: f64,
	im: f64,
) -> (Vec<f64>, Vec<f64>) {
	let n = t.num_rows();
	let mut yr = vec![0.0f64; n];
	let mut yi = vec![0.0f64; n];
	yr[p] = 1.0;
	yi[p + 1] = 1.0;

	for i in (0..p).rev() {
		let mut accr = 0.0f64;
		let mut acci = 0.0f64;
		for j in (i + 1)..n {
			let tij = t[(i, j)];
			accr -= tij * yr[j];
			acci -= tij * yi[j];
		}
		let dre = t[(i, i)] - re;
		let det = dre * dre + im * im;
		if det > 1e-28 {
			yr[i] = (dre * accr + im * acci) / det;
			yi[i] = (dre * acci - im * accr) / det;
		}
	}

	let norm = yr
		.iter()
		.chain(yi.iter())
		.map(|x| x * x)
		.sum::<f64>()
		.sqrt()
		.max(f64::EPSILON);
	yr.iter_mut().for_each(|x| *x /= norm);
	yi.iter_mut().for_each(|x| *x /= norm);
	(yr, yi)
}

fn hessenberg_reduce_with_z(a: MatrixRef<f64>) -> (Box<[f64]>, Box<[f64]>) {
	let n = a.num_rows();
	let mut h_box = a.to_boxed_slice();
	let mut z_box = vec![0.0f64; n * n].into_boxed_slice();

	let mut h = MatrixMut::from_slice(&mut h_box, n, n);
	let mut z = MatrixMut::from_slice(&mut z_box, n, n);

	z.reborrow().identity();

	for k in 0..n.saturating_sub(2) {
		let mut norm_sq = 0.0f64;
		for i in (k + 1)..n {
			norm_sq += h[(i, k)] * h[(i, k)];
		}
		let norm = norm_sq.sqrt();
		if norm < f64::EPSILON {
			continue;
		}

		let mut v = vec![0.0f64; n];
		for i in (k + 1)..n {
			v[i] = h[(i, k)];
		}
		let sign = if v[k + 1] >= 0.0 { 1.0 } else { -1.0 };
		v[k + 1] += sign * norm;

		let vv: f64 = ((k + 1)..n).map(|i| v[i] * v[i]).sum();
		if vv < f64::EPSILON {
			continue;
		}
		let two_over_vv = 2.0 / vv;

		for j in k..n {
			let mut dot = 0.0f64;
			for i in (k + 1)..n {
				dot += v[i] * h[(i, j)];
			}
			let f = two_over_vv * dot;
			for i in (k + 1)..n {
				h[(i, j)] -= f * v[i];
			}
		}

		for i in 0..n {
			let mut dot = 0.0f64;
			for j in (k + 1)..n {
				dot += h[(i, j)] * v[j];
			}
			let f = two_over_vv * dot;
			for j in (k + 1)..n {
				h[(i, j)] -= f * v[j];
			}
		}

		for i in 0..n {
			let mut dot = 0.0f64;
			for j in (k + 1)..n {
				dot += z[(i, j)] * v[j];
			}
			let f = two_over_vv * dot;
			for j in (k + 1)..n {
				z[(i, j)] -= f * v[j];
			}
		}
	}
	(h_box, z_box)
}

fn francis_qr(mut h: MatrixMut<f64>, mut z: MatrixMut<f64>) {
	let n = h.num_rows();
	let mut zt_box = vec![0.0f64; n * n].into_boxed_slice();
	let mut zt = MatrixMut::from_slice(&mut zt_box, n, n);

	for i in 0..n {
		for j in 0..n {
			zt[(j, i)] = z[(i, j)];
		}
	}

	let mut active_end = n;
	let mut iter = 0;
	let max_iter = 100 * n;

	while active_end > 1 && iter < max_iter {
		let active_start =
			find_deflation_point(h.reborrow(), active_end);

		if active_end - active_start <= 2 {
			if active_end - active_start == 2 {
				let p = active_start;
				let a = h[(p, p)];
				let b = h[(p, p + 1)];
				let c = h[(p + 1, p)];
				let d = h[(p + 1, p + 1)];
				if c.abs() > f64::EPSILON * (a.abs() + d.abs())
				{
					let tr = a + d;
					let det = a * d - b * c;
					let disc = tr * tr - 4.0 * det;
					if disc >= 0.0 {
						standardize_2x2_real(
							h.reborrow(),
							zt.reborrow(),
							p,
						);
					}
				}
			}
			active_end = active_start;
			continue;
		}

		let (sigma_sum, sigma_prod) = if iter % 10 == 9 {
			let s1 = h[(active_end - 1, active_end - 2)].abs();
			let s2 = if active_end >= 3 {
				h[(active_end - 2, active_end - 3)].abs()
			} else {
				0.0
			};
			let t = s1 + s2;
			(1.5 * t, 0.8125 * t * t)
		} else {
			wilkinson_shift(h.reborrow(), active_end)
		};

		{
			francis_double_step(
				h.reborrow(),
				zt.reborrow(),
				active_start,
				active_end,
				sigma_sum,
				sigma_prod,
			);
		}
		iter += 1;
	}

	let zt = MatrixRef::from_slice(&zt_box, n, n);
	for i in 0..n {
		for j in 0..n {
			z[(i, j)] = zt[(j, i)];
		}
	}
}

fn standardize_2x2_real(
	mut h: MatrixMut<f64>,
	mut zt: MatrixMut<f64>,
	p: usize,
) {
	let n = h.num_rows();
	let a = h[(p, p)];
	let b = h[(p, p + 1)];
	let c = h[(p + 1, p)];
	let d = h[(p + 1, p + 1)];
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

	for j in 0..n {
		let t0 = cs * h[(p, j)] + sn * h[(p + 1, j)];
		let t1 = -sn * h[(p, j)] + cs * h[(p + 1, j)];
		h[(p, j)] = t0;
		h[(p + 1, j)] = t1;
	}
	for i in 0..n {
		let t0 = cs * h[(i, p)] + sn * h[(i, p + 1)];
		let t1 = -sn * h[(i, p)] + cs * h[(i, p + 1)];
		h[(i, p)] = t0;
		h[(i, p + 1)] = t1;
	}
	h[(p + 1, p)] = 0.0;
	for j in 0..n {
		let t0 = cs * zt[(p, j)] + sn * zt[(p + 1, j)];
		let t1 = -sn * zt[(p, j)] + cs * zt[(p + 1, j)];
		zt[(p, j)] = t0;
		zt[(p + 1, j)] = t1;
	}
}

fn find_deflation_point(mut h: MatrixMut<f64>, active_end: usize) -> usize {
	for i in (0..active_end.saturating_sub(1)).rev() {
		let sub = h[(i + 1, i)].abs();
		let scale = h[(i, i)].abs() + h[(i + 1, i + 1)].abs();
		if sub <= f64::EPSILON * scale {
			h[(i + 1, i)] = 0.0;
			return i + 1;
		}
	}
	0
}

fn wilkinson_shift(h: MatrixMut<f64>, end: usize) -> (f64, f64) {
	let a = h[(end - 2, end - 2)];
	let d = h[(end - 1, end - 1)];
	(a + d, a * d - h[(end - 2, end - 1)] * h[(end - 1, end - 2)])
}

fn francis_double_step(
	mut h: MatrixMut<f64>,
	mut zt: MatrixMut<f64>,
	start: usize,
	end: usize,
	shift_sum: f64,
	shift_prod: f64,
) {
	let n = h.num_rows();
	let (s, e) = (start, end);
	let sub = h[(s + 1, s)];
	let p0 = h[(s, s)] * h[(s, s)] + h[(s, s + 1)] * sub
		- shift_sum * h[(s, s)]
		+ shift_prod;
	let p1 = sub * (h[(s, s)] + h[(s + 1, s + 1)] - shift_sum);
	let p2 = h[(s + 2, s + 1)] * sub;
	let mut x = [p0, p1, p2];

	for k in 0..(e - s - 2) {
		let r = s + k;
		let mut v = x;
		let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
		if norm < f64::EPSILON {
			if k + 1 < e - s - 2 {
				x = [
					h[(r + 1, r)],
					h[(r + 2, r)],
					h[(r + 3, r)],
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
			let dot = v[0] * h[(r, j)]
				+ v[1] * h[(r + 1, j)] + v[2] * h[(r + 2, j)];
			let f = two_over_vv * dot;
			h[(r, j)] -= f * v[0];
			h[(r + 1, j)] -= f * v[1];
			h[(r + 2, j)] -= f * v[2];
		}
		for i in 0..(r + 4).min(e) {
			let dot = v[0] * h[(i, r)]
				+ v[1] * h[(i, r + 1)] + v[2] * h[(i, r + 2)];
			let f = two_over_vv * dot;
			h[(i, r)] -= f * v[0];
			h[(i, r + 1)] -= f * v[1];
			h[(i, r + 2)] -= f * v[2];
		}
		for j in 0..n {
			let dot = v[0] * zt[(r, j)]
				+ v[1] * zt[(r + 1, j)] + v[2] * zt
				[(r + 2, j)];
			let f = two_over_vv * dot;
			zt[(r, j)] -= f * v[0];
			zt[(r + 1, j)] -= f * v[1];
			zt[(r + 2, j)] -= f * v[2];
		}
		if k + 1 < e - s - 2 {
			x = [h[(r + 1, r)], h[(r + 2, r)], h[(r + 3, r)]];
		}
	}

	let (rt, rb, cf) = (e - 2, e - 1, e - 3);
	let (x0, v1) = (h[(rt, cf)], h[(rb, cf)]);
	let n2 = (x0 * x0 + v1 * v1).sqrt();
	if n2 > f64::EPSILON {
		let v0 = x0 + (if x0 >= 0.0 { 1.0 } else { -1.0 }) * n2;
		let two_over_vv = 2.0 / (v0 * v0 + v1 * v1);
		for j in cf..n {
			let dot = v0 * h[(rt, j)] + v1 * h[(rb, j)];
			let f = two_over_vv * dot;
			h[(rt, j)] -= f * v0;
			h[(rb, j)] -= f * v1;
		}
		for i in 0..e {
			let dot = v0 * h[(i, rt)] + v1 * h[(i, rb)];
			let f = two_over_vv * dot;
			h[(i, rt)] -= f * v0;
			h[(i, rb)] -= f * v1;
		}
		for j in 0..n {
			let dot = v0 * zt[(rt, j)] + v1 * zt[(rb, j)];
			let f = two_over_vv * dot;
			zt[(rt, j)] -= f * v0;
			zt[(rb, j)] -= f * v1;
		}
	}
}

fn tung_tung_tung_schur(t: MatrixRef<f64>, re: &mut [f64], im: &mut [f64]) {
	let n = t.num_rows();
	let mut i = 0;
	while i < n {
		let is_2x2 = i + 1 < n
			&& t[(i + 1, i)].abs()
				> f64::EPSILON
					* (t[(i, i)].abs()
						+ t[(i + 1, i + 1)].abs());
		if is_2x2 {
			let (a, d) = (t[(i, i)], t[(i + 1, i + 1)]);
			let det = a * d - t[(i, i + 1)] * t[(i + 1, i)];
			let tr = a + d;
			let disc = tr * tr - 4.0 * det;
			if disc >= 0.0 {
				re[i] = t[(i, i)];
				re[i + 1] = t[(i + 1, i + 1)];
			} else {
				re[i] = tr * 0.5;
				re[i + 1] = tr * 0.5;
				im[i] = (-disc).sqrt() * 0.5;
				im[i + 1] = -im[i];
			}
			i += 2;
		} else {
			re[i] = t[(i, i)];
			i += 1;
		}
	}
}
