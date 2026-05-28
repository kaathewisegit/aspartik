#![allow(clippy::all)]
#![allow(clippy::branches_sharing_code)]
//! This is an LLM rewrite of the DefaultEigenSystem implementation from BEAST.
//! Hopefully, it is only here as a stopgap measure until I finish working on
//! the new iteration of linalg.  But that might not happen any time soon.

pub struct EigenDecomposition {
	flat_evec: Vec<f64>,
	flat_ievc: Vec<f64>,
	eval: Vec<f64>,
}

impl EigenDecomposition {
	fn new(
		flat_evec: Vec<f64>,
		flat_ievc: Vec<f64>,
		eval: Vec<f64>,
	) -> Self {
		Self {
			flat_evec,
			flat_ievc,
			eval,
		}
	}

	pub fn get_eigen_vectors(&self) -> &[f64] {
		&self.flat_evec
	}

	pub fn get_inverse_eigen_vectors(&self) -> &[f64] {
		&self.flat_ievc
	}

	pub fn get_eigen_values(&self) -> &[f64] {
		&self.eval
	}
}

const MACHINE_EPSILON: f64 = f64::EPSILON;

pub struct DefaultEigenSystem {
	state_count: usize,
	eval: Vec<f64>,
	evec: Vec<Vec<f64>>,
	ievc: Vec<Vec<f64>>,
	ordr: Vec<usize>,
	evali: Vec<f64>,
}

impl DefaultEigenSystem {
	pub fn new(state_count: usize) -> Self {
		let ordr = vec![0usize; state_count];
		let evali = vec![0.0; state_count];
		Self {
			state_count,
			eval: Vec::new(),
			evec: Vec::new(),
			ievc: Vec::new(),
			ordr,
			evali,
		}
	}

	pub fn decompose_matrix(
		&mut self,
		q_matrix: &mut Vec<Vec<f64>>,
	) -> Result<EigenDecomposition, String> {
		let n = self.state_count;

		self.eval = vec![0.0; n];
		self.evec = vec![vec![0.0; n]; n];
		self.ievc = vec![vec![0.0; n]; n];

		Self::elmhes(q_matrix, &mut self.ordr, n);
		Self::eltran(q_matrix, &mut self.evec, &self.ordr, n);
		Self::hqr2(
			n,
			1,
			n,
			q_matrix,
			&mut self.evec,
			&mut self.eval,
			&mut self.evali,
		)?;
		Self::luinverse(&self.evec, &mut self.ievc, n)?;

		let mut flat_evec = vec![0.0; n * n];
		let mut flat_ievc = vec![0.0; n * n];

		for i in 0..n {
			flat_evec[i * n..(i + 1) * n]
				.copy_from_slice(&self.evec[i]);
			flat_ievc[i * n..(i + 1) * n]
				.copy_from_slice(&self.ievc[i]);
		}

		Ok(EigenDecomposition::new(
			flat_evec,
			flat_ievc,
			self.eval.clone(),
		))
	}

	pub fn compute_exponential(
		&self,
		eigen: Option<&EigenDecomposition>,
		distance: f64,
		i: usize,
		j: usize,
	) -> f64 {
		if eigen.is_none() {
			return 0.0;
		}
		let eigen = eigen.unwrap();

		let n = self.state_count;
		let evec = eigen.get_eigen_vectors();
		let eval = eigen.get_eigen_values();
		let ievc = eigen.get_inverse_eigen_vectors();

		let mut temp = 0.0;
		for k in 0..n {
			temp += evec[i * n + k]
				* (distance * eval[k]).exp() * ievc[k * n + j];
		}
		temp.abs()
	}

	pub fn compute_exponential_matrix(
		&self,
		eigen: Option<&EigenDecomposition>,
		distance: f64,
		matrix: &mut [f64],
	) {
		let mut temp: f64;

		if eigen.is_none() {
			matrix.fill(0.0);
			return;
		}
		let eigen = eigen.unwrap();

		let n = self.state_count;
		let evec = eigen.get_eigen_vectors();
		let ievc = eigen.get_inverse_eigen_vectors();
		let eval = eigen.get_eigen_values();

		let mut iexp = vec![vec![0.0; n]; n];
		for i in 0..n {
			temp = (distance * eval[i]).exp();
			for j in 0..n {
				iexp[i][j] = ievc[i * n + j] * temp;
			}
		}

		let mut u = 0;
		for i in 0..n {
			for j in 0..n {
				temp = 0.0;
				for k in 0..n {
					temp += evec[i * n + k] * iexp[k][j];
				}
				matrix[u] = temp.abs();
				u += 1;
			}
		}
	}

	#[allow(clippy::many_single_char_names)]
	fn elmhes(a: &mut [Vec<f64>], ordr: &mut [usize], n: usize) {
		let mut m: usize;
		let mut j: usize;
		let mut i: usize;
		let mut y: f64;
		let mut x: f64;

		for i_val in 0..n {
			ordr[i_val] = 0;
		}
		m = 2;
		while m < n {
			x = 0.0;
			i = m;
			j = m;
			while j <= n {
				if a[j - 1][m - 2].abs() > x.abs() {
					x = a[j - 1][m - 2];
					i = j;
				}
				j += 1;
			}
			ordr[m - 1] = i;
			if i != m {
				j = m - 2;
				while j < n {
					y = a[i - 1][j];
					a[i - 1][j] = a[m - 1][j];
					a[m - 1][j] = y;
					j += 1;
				}
				j = 0;
				while j < n {
					y = a[j][i - 1];
					a[j][i - 1] = a[j][m - 1];
					a[j][m - 1] = y;
					j += 1;
				}
			}
			if x != 0.0 {
				i = m;
				while i < n {
					y = a[i][m - 2];
					if y != 0.0 {
						y /= x;
						a[i][m - 2] = y;
						j = m - 1;
						while j < n {
							a[i][j] -=
								y * a[m - 1][j];
							j += 1;
						}
						j = 0;
						while j < n {
							a[j][m - 1] +=
								y * a[j][i];
							j += 1;
						}
					}
					i += 1;
				}
			}
			m += 1;
		}
	}

	fn mcdiv(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
		let s = br.abs() + bi.abs();
		let ars = ar / s;
		let ais = ai / s;
		let brs = br / s;
		let bis = bi / s;
		let s = brs * brs + bis * bis;
		let cr = (ars * brs + ais * bis) / s;
		let ci = (ais * brs - ars * bis) / s;
		(cr, ci)
	}

	#[allow(
		unused_assignments,
		unused_mut,
		clippy::too_many_arguments,
		clippy::many_single_char_names
	)]
	fn hqr2(
		n: usize,
		low: usize,
		hgh: usize,
		h: &mut [Vec<f64>],
		zz: &mut [Vec<f64>],
		wr: &mut [f64],
		wi: &mut [f64],
	) -> Result<(), String> {
		let mut i: usize = 0;
		let mut j: usize = 0;
		let mut k: usize = 0;
		let mut l: usize = 0;
		let mut m: usize = 0;
		let mut en: usize;
		let mut na: usize;
		let mut itn: usize;
		let mut its: usize;
		let mut p: f64 = 0.0;
		let mut q: f64 = 0.0;
		let mut r: f64 = 0.0;
		let mut s: f64 = 0.0;
		let mut t: f64 = 0.0;
		let mut w: f64 = 0.0;
		let mut x: f64 = 0.0;
		let mut y: f64 = 0.0;
		let mut ra: f64 = 0.0;
		let mut sa: f64 = 0.0;
		let mut vi: f64 = 0.0;
		let mut vr: f64 = 0.0;
		let mut z: f64 = 0.0;
		let mut norm: f64;
		let mut tst1: f64;
		let mut tst2: f64;
		let mut not_last: bool;

		norm = 0.0;
		k = 1;
		for i_val in 0..n {
			j = k - 1;
			while j < n {
				norm += h[i_val][j].abs();
				j += 1;
			}
			k = i_val + 1;
			if i_val + 1 < low || i_val + 1 > hgh {
				wr[i_val] = h[i_val][i_val];
				wi[i_val] = 0.0;
			}
		}
		en = hgh;
		t = 0.0;
		itn = n * 30;
		'outer: while en >= low {
			its = 0;
			na = en - 1;
			loop {
				let mut full_loop = true;
				l = en;
				while l > low {
					s = h[l - 2][l - 2].abs()
						+ h[l - 1][l - 1].abs();
					if s == 0.0 {
						s = norm;
					}
					tst1 = s;
					tst2 = tst1 + h[l - 1][l - 2].abs();
					if tst2 == tst1 {
						full_loop = false;
						break;
					}
					l -= 1;
				}
				if full_loop {
					l = low;
				}

				x = h[en - 1][en - 1];
				if l == en || l == na {
					break;
				}
				if itn == 0 {
					return Err(
						"Eigenvalues not converged"
							.to_string(),
					);
				}

				y = h[na - 1][na - 1];
				w = h[en - 1][na - 1] * h[na - 1][en - 1];

				if its == 10 || its == 20 {
					t += x;
					i = low - 1;
					while i < en {
						h[i][i] -= x;
						i += 1;
					}
					s = h[en - 1][na - 1].abs()
						+ h[na - 1][en - 3].abs();
					x = 0.75 * s;
					y = x;
					w = -0.4375 * s * s;
				}
				its += 1;
				itn -= 1;

				m = en - 2;
				while m >= l {
					z = h[m - 1][m - 1];
					r = x - z;
					s = y - z;
					p = (r * s - w) / h[m][m - 1]
						+ h[m - 1][m];
					q = h[m][m] - z - r - s;
					r = h[m + 1][m];
					s = p.abs() + q.abs() + r.abs();
					p /= s;
					q /= s;
					r /= s;
					if m == l {
						break;
					}
					tst1 = p.abs()
						* (h[m - 2][m - 2].abs()
							+ z.abs() + h[m][m].abs());
					tst2 = tst1 + h[m - 1][m - 2].abs()
						* (q.abs() + r.abs());
					if tst2 == tst1 {
						break;
					}
					m -= 1;
				}

				i = m + 2;
				while i <= en {
					h[i - 1][i - 3] = 0.0;
					if i != m + 2 {
						h[i - 1][i - 4] = 0.0;
					}
					i += 1;
				}
				k = m;
				while k <= na {
					not_last = k != na;
					if k != m {
						p = h[k - 1][k - 2];
						q = h[k][k - 2];
						r = 0.0;
						if not_last {
							r = h[k + 1][k - 2];
						}
						x = p.abs() + q.abs() + r.abs();
						if x != 0.0 {
							p /= x;
							q /= x;
							r /= x;
						}
					}
					if x != 0.0 {
						if p < 0.0 {
							s = -(p * p
								+ q * q + r * r)
								.sqrt();
						} else {
							s = (p * p
								+ q * q + r * r)
								.sqrt();
						}
						if k != m {
							h[k - 1][k - 2] =
								-s * x;
						} else if l != m {
							h[k - 1][k - 2] = -h
								[k - 1][k - 2];
						}
						p += s;
						x = p / s;
						y = q / s;
						z = r / s;
						q /= p;
						r /= p;
						if !not_last {
							j = k - 1;
							while j < n {
								p = h[k - 1][j] + q * h[k][j];
								h[k - 1][j] -=
									p * x;
								h[k][j] -=
									p * y;
								j += 1;
							}
							j = if en < k + 3 {
								en
							} else {
								k + 3
							};
							i = 0;
							while i < j {
								p = x * h[i][k - 1] + y * h[i][k];
								h[i][k - 1] -=
									p;
								h[i][k] -=
									p * q;
								i += 1;
							}
							i = low - 1;
							while i < hgh {
								p = x * zz[i][k - 1] + y * zz[i][k];
								zz[i][k - 1] -=
									p;
								zz[i][k] -=
									p * q;
								i += 1;
							}
						} else {
							j = k - 1;
							while j < n {
								p = h[k - 1][j] + q * h[k][j] + r * h[k + 1][j];
								h[k - 1][j] -=
									p * x;
								h[k][j] -=
									p * y;
								h[k + 1][j] -=
									p * z;
								j += 1;
							}
							j = if en < k + 3 {
								en
							} else {
								k + 3
							};
							i = 0;
							while i < j {
								p = x * h[i][k - 1]
                                    + y * h[i][k]
                                    + z * h[i][k + 1];
								h[i][k - 1] -=
									p;
								h[i][k] -=
									p * q;
								h[i][k + 1] -=
									p * r;
								i += 1;
							}
							i = low - 1;
							while i < hgh {
								p = x * zz[i][k - 1]
                                    + y * zz[i][k]
                                    + z * zz[i][k + 1];
								zz[i][k - 1] -=
									p;
								zz[i][k] -=
									p * q;
								zz[i][k + 1] -=
									p * r;
								i += 1;
							}
						}
					}
					k += 1;
				}
			}

			if l == en {
				h[en - 1][en - 1] = x + t;
				wr[en - 1] = h[en - 1][en - 1];
				wi[en - 1] = 0.0;
				en = na;
				continue 'outer;
			}

			y = h[na - 1][na - 1];
			w = h[en - 1][na - 1] * h[na - 1][en - 1];
			p = (y - x) / 2.0;
			q = p * p + w;
			z = q.abs().sqrt();
			h[en - 1][en - 1] = x + t;
			x = h[en - 1][en - 1];
			h[na - 1][na - 1] = y + t;
			if q >= 0.0 {
				if p < 0.0 {
					z = p - z.abs();
				} else {
					z = p + z.abs();
				}
				wr[na - 1] = x + z;
				wr[en - 1] = wr[na - 1];
				if z != 0.0 {
					wr[en - 1] = x - w / z;
				}
				wi[na - 1] = 0.0;
				wi[en - 1] = 0.0;
				x = h[en - 1][na - 1];
				s = x.abs() + z.abs();
				p = x / s;
				q = z / s;
				r = (p * p + q * q).sqrt();
				p /= r;
				q /= r;
				j = na - 1;
				while j < n {
					z = h[na - 1][j];
					h[na - 1][j] = q * z + p * h[en - 1][j];
					h[en - 1][j] = q * h[en - 1][j] - p * z;
					j += 1;
				}
				i = 0;
				while i < en {
					z = h[i][na - 1];
					h[i][na - 1] = q * z + p * h[i][en - 1];
					h[i][en - 1] = q * h[i][en - 1] - p * z;
					i += 1;
				}
				i = low - 1;
				while i < hgh {
					z = zz[i][na - 1];
					zz[i][na - 1] =
						q * z + p * zz[i][en - 1];
					zz[i][en - 1] =
						q * zz[i][en - 1] - p * z;
					i += 1;
				}
			} else {
				wr[na - 1] = x + p;
				wr[en - 1] = x + p;
				wi[na - 1] = z;
				wi[en - 1] = -z;
			}
			en -= 2;
		}

		if norm != 0.0 {
			en = n;
			while en >= 1 {
				p = wr[en - 1];
				q = wi[en - 1];
				na = en - 1;
				if q == 0.0 {
					m = en;
					h[en - 1][en - 1] = 1.0;
					if na != 0 {
						i = en - 2;
						loop {
							w = h[i][i] - p;
							r = 0.0;
							j = m - 1;
							while j < en {
								r += h[i][j] * h[j][en - 1];
								j += 1;
							}
							if wi[i] < 0.0 {
								z = w;
								s = r;
							} else {
								m = i + 1;
								if wi[i] == 0.0
								{
									t = w;
									if t == 0.0 {
                                        tst1 = norm;
                                        t = tst1;
                                        loop {
                                            t = 0.01 * t;
                                            tst2 = norm + t;
                                            if !(tst2 > tst1) {
                                                break;
                                            }
                                        }
                                    }
									h[i][en - 1] = -(r / t);
								} else {
									x = h[i][i + 1];
									y = h[i + 1][i];
									q = (wr[i] - p) * (wr[i] - p)
                                        + wi[i] * wi[i];
									t = (x * s - z * r) / q;
									h[i][en - 1] = t;
									if x.abs() > z.abs() {
                                        h[i + 1][en - 1] = (-r - w * t) / x;
                                    } else {
                                        h[i + 1][en - 1] = (-s - y * t) / z;
                                    }
								}
								t = h[i][en
									- 1]
								.abs();
								if t != 0.0 {
									tst1 = t;
									tst2 = tst1 + 1.0 / tst1;
									if tst2 <= tst1 {
                                        j = i;
                                        while j < en {
                                            h[j][en - 1] /= t;
                                            j += 1;
                                        }
                                    }
								}
							}
							if i == 0 {
								break;
							}
							i -= 1;
						}
					}
				} else if q > 0.0 {
					m = na;
					if h[en - 1][na - 1].abs()
						> h[na - 1][en - 1].abs()
					{
						h[na - 1][na - 1] =
							q / h[en - 1][na - 1];
						h[na - 1][en - 1] = (p - h
							[en - 1][en - 1])
							/ h[en - 1][na - 1];
					} else {
						let (cr, ci) = Self::mcdiv(
							0.0,
							-h[na - 1][en - 1],
							h[na - 1][na - 1] - p,
							q,
						);
						h[na - 1][na - 1] = cr;
						h[na - 1][en - 1] = ci;
					}
					h[en - 1][na - 1] = 0.0;
					h[en - 1][en - 1] = 1.0;
					if en != 2 {
						i = en - 3;
						loop {
							w = h[i][i] - p;
							ra = 0.0;
							sa = 0.0;
							j = m - 1;
							while j < en {
								ra += h[i][j] * h[j][na - 1];
								sa += h[i][j] * h[j][en - 1];
								j += 1;
							}
							if wi[i] < 0.0 {
								z = w;
								r = ra;
								s = sa;
							} else {
								m = i + 1;
								if wi[i] == 0.0
								{
									let (cr, ci) = Self::mcdiv(-ra, -sa, w, q);
									h[i][na - 1] = cr;
									h[i][en - 1] = ci;
								} else {
									x = h[i][i + 1];
									y = h[i + 1][i];
									vr = (wr[i] - p) * (wr[i] - p);
									vr = vr + wi[i] * wi[i] - q * q;
									vi = (wr[i] - p) * 2.0 * q;
									if vr == 0.0 && vi == 0.0 {
                                        tst1 = norm
                                            * (w.abs()
                                                + q.abs()
                                                + x.abs()
                                                + y.abs()
                                                + z.abs());
                                        vr = tst1;
                                        loop {
                                            vr = 0.01 * vr;
                                            tst2 = tst1 + vr;
                                            if !(tst2 > tst1) {
                                                break;
                                            }
                                        }
                                    }
									let (cr, ci) = Self::mcdiv(
                                        x * r - z * ra + q * sa,
                                        x * s - z * sa - q * ra,
                                        vr,
                                        vi,
                                    );
									h[i][na - 1] = cr;
									h[i][en - 1] = ci;
									if x.abs() > z.abs() + q.abs() {
                                        h[i + 1][na - 1] = (q * h[i][en - 1]
                                            - w * h[i][na - 1]
                                            - ra)
                                            / x;
                                        h[i + 1][en - 1] = (-sa
                                            - w * h[i][en - 1]
                                            - q * h[i][na - 1])
                                            / x;
                                    } else {
                                        let (cr, ci) = Self::mcdiv(
                                            -r - y * h[i][na - 1],
                                            -s - y * h[i][en - 1],
                                            z,
                                            q,
                                        );
                                        h[i + 1][na - 1] = cr;
                                        h[i + 1][en - 1] = ci;
                                    }
								}
								t = if h[i][na - 1].abs() > h[i][en - 1].abs() {
                                    h[i][na - 1].abs()
                                } else {
                                    h[i][en - 1].abs()
                                };
								if t != 0.0 {
									tst1 = t;
									tst2 = tst1 + 1.0 / tst1;
									if tst2 <= tst1 {
                                        j = i;
                                        while j < en {
                                            h[j][na - 1] /= t;
                                            h[j][en - 1] /= t;
                                            j += 1;
                                        }
                                    }
								}
							}
							if i == 0 {
								break;
							}
							i -= 1;
						}
					}
				}
				en -= 1;
			}

			i = 0;
			while i < n {
				if i + 1 < low || i + 1 > hgh {
					j = i;
					while j < n {
						zz[i][j] = h[i][j];
						j += 1;
					}
				}
				i += 1;
			}

			j = n;
			loop {
				if j == 0 {
					break;
				}
				j -= 1;
				if j < low - 1 {
					continue;
				}
				m = if j + 1 < hgh { j + 1 } else { hgh };
				i = low - 1;
				while i < hgh {
					z = 0.0;
					k = low - 1;
					while k < m {
						z += zz[i][k] * h[k][j];
						k += 1;
					}
					zz[i][j] = z;
					i += 1;
				}
			}
		}

		Ok(())
	}

	fn eltran(
		a: &[Vec<f64>],
		zz: &mut [Vec<f64>],
		ordr: &[usize],
		n: usize,
	) {
		let mut i: usize;
		let mut j: usize;
		let mut m: usize;

		i = 0;
		while i < n {
			j = i + 1;
			while j < n {
				zz[i][j] = 0.0;
				zz[j][i] = 0.0;
				j += 1;
			}
			zz[i][i] = 1.0;
			i += 1;
		}
		if n <= 2 {
			return;
		}
		m = n - 1;
		while m >= 2 {
			i = m;
			while i < n {
				zz[i][m - 1] = a[i][m - 2];
				i += 1;
			}
			i = ordr[m - 1];
			if i != m {
				j = m - 1;
				while j < n {
					zz[m - 1][j] = zz[i - 1][j];
					zz[i - 1][j] = 0.0;
					j += 1;
				}
				zz[i - 1][m - 1] = 1.0;
			}
			m -= 1;
		}
	}

	fn luinverse(
		inmat: &[Vec<f64>],
		imtrx: &mut [Vec<f64>],
		size: usize,
	) -> Result<(), String> {
		let mut i: usize;
		let mut j: usize;
		let mut k: usize;
		let mut l: isize;
		let mut maxi: usize = 0;
		let mut idx: usize;
		let mut ix: usize;
		let mut jx: usize;
		let mut sum: f64;
		let mut tmp: f64;
		let mut maxb: f64;
		let mut aw: f64;

		let mut index = vec![0usize; size];
		let mut omtrx = inmat.to_vec();
		let mut wk = vec![0.0; size];

		aw = 1.0;
		i = 0;
		while i < size {
			maxb = 0.0;
			j = 0;
			while j < size {
				if omtrx[i][j].abs() > maxb {
					maxb = omtrx[i][j].abs();
				}
				j += 1;
			}
			if maxb == 0.0 {
				return Err("Singular matrix".to_string());
			}
			wk[i] = 1.0 / maxb;
			i += 1;
		}
		j = 0;
		while j < size {
			i = 0;
			while i < j {
				sum = omtrx[i][j];
				k = 0;
				while k < i {
					sum -= omtrx[i][k] * omtrx[k][j];
					k += 1;
				}
				omtrx[i][j] = sum;
				i += 1;
			}
			maxb = 0.0;
			i = j;
			while i < size {
				sum = omtrx[i][j];
				k = 0;
				while k < j {
					sum -= omtrx[i][k] * omtrx[k][j];
					k += 1;
				}
				omtrx[i][j] = sum;
				tmp = wk[i] * sum.abs();
				if tmp >= maxb {
					maxb = tmp;
					maxi = i;
				}
				i += 1;
			}
			if j != maxi {
				k = 0;
				while k < size {
					tmp = omtrx[maxi][k];
					omtrx[maxi][k] = omtrx[j][k];
					omtrx[j][k] = tmp;
					k += 1;
				}
				aw = -aw;
				wk[maxi] = wk[j];
			}
			index[j] = maxi;
			if omtrx[j][j] == 0.0 {
				omtrx[j][j] = MACHINE_EPSILON;
			}
			if j != size - 1 {
				tmp = 1.0 / omtrx[j][j];
				i = j + 1;
				while i < size {
					omtrx[i][j] *= tmp;
					i += 1;
				}
			}
			j += 1;
		}
		jx = 0;
		while jx < size {
			ix = 0;
			while ix < size {
				wk[ix] = 0.0;
				ix += 1;
			}
			wk[jx] = 1.0;
			l = -1;
			i = 0;
			while i < size {
				idx = index[i];
				sum = wk[idx];
				wk[idx] = wk[i];
				if l != -1 {
					j = l as usize;
					while j < i {
						sum -= omtrx[i][j] * wk[j];
						j += 1;
					}
				} else if sum != 0.0 {
					l = i as isize;
				}
				wk[i] = sum;
				i += 1;
			}
			i = size - 1;
			loop {
				sum = wk[i];
				j = i + 1;
				while j < size {
					sum -= omtrx[i][j] * wk[j];
					j += 1;
				}
				wk[i] = sum / omtrx[i][i];
				if i == 0 {
					break;
				}
				i -= 1;
			}
			ix = 0;
			while ix < size {
				imtrx[ix][jx] = wk[ix];
				ix += 1;
			}
			jx += 1;
		}

		Ok(())
	}
}
