#![allow(clippy::all)]
#![allow(clippy::branches_sharing_code)]
//! This is an LLM rewrite of the DefaultEigenSystem implementation from BEAST.
//! Hopefully, it is only here as a stopgap measure until I finish working on
//! the new iteration of linalg.  But that might not happen any time soon.

use computare_linalg::{Matrix, RowMatrix};

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
	evec: RowMatrix<f64>,
	ievc: RowMatrix<f64>,
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
			evec: RowMatrix::empty(),
			ievc: RowMatrix::empty(),
			ordr,
			evali,
		}
	}

	pub fn decompose_matrix(
		&mut self,
		q_matrix: &mut RowMatrix<f64>,
	) -> Result<EigenDecomposition, String> {
		let n = self.state_count;

		self.eval = vec![0.0; n];
		self.evec = RowMatrix::zeroed(n, n);
		self.ievc = RowMatrix::zeroed(n, n);

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

		flat_evec.copy_from_slice(self.evec.as_slice());
		flat_ievc.copy_from_slice(self.ievc.as_slice());

		Ok(EigenDecomposition::new(
			flat_evec,
			flat_ievc,
			self.eval.clone(),
		))
	}

	#[allow(clippy::many_single_char_names)]
	fn elmhes(a: &mut RowMatrix<f64>, ordr: &mut [usize], n: usize) {
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
				if a.at(j - 1, m - 2).abs() > x.abs() {
					x = *a.at(j - 1, m - 2);
					i = j;
				}
				j += 1;
			}
			ordr[m - 1] = i;
			if i != m {
				j = m - 2;
				while j < n {
					y = *a.at(i - 1, j);
					*a.at_mut(i - 1, j) = *a.at(m - 1, j);
					*a.at_mut(m - 1, j) = y;
					j += 1;
				}
				j = 0;
				while j < n {
					y = *a.at(j, i - 1);
					*a.at_mut(j, i - 1) = *a.at(j, m - 1);
					*a.at_mut(j, m - 1) = y;
					j += 1;
				}
			}
			if x != 0.0 {
				i = m;
				while i < n {
					y = *a.at(i, m - 2);
					if y != 0.0 {
						y /= x;
						*a.at_mut(i, m - 2) = y;
						j = m - 1;
						while j < n {
							*a.at_mut(i, j) -= y
								* *a.at(
									m - 1,
									j,
								);
							j += 1;
						}
						j = 0;
						while j < n {
							*a.at_mut(j, m - 1) +=
								y * *a.at(j, i);
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
		h: &mut RowMatrix<f64>,
		zz: &mut RowMatrix<f64>,
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
				norm += h.at(i_val, j).abs();
				j += 1;
			}
			k = i_val + 1;
			if i_val + 1 < low || i_val + 1 > hgh {
				wr[i_val] = *h.at(i_val, i_val);
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
					s = h.at(l - 2, l - 2).abs()
						+ h.at(l - 1, l - 1).abs();
					if s == 0.0 {
						s = norm;
					}
					tst1 = s;
					tst2 = tst1 + h.at(l - 1, l - 2).abs();
					if tst2 == tst1 {
						full_loop = false;
						break;
					}
					l -= 1;
				}
				if full_loop {
					l = low;
				}

				x = *h.at(en - 1, en - 1);
				if l == en || l == na {
					break;
				}
				if itn == 0 {
					return Err(
						"Eigenvalues not converged"
							.to_string(),
					);
				}

				y = *h.at(na - 1, na - 1);
				w = *h.at(en - 1, na - 1)
					* *h.at(na - 1, en - 1);

				if its == 10 || its == 20 {
					t += x;
					i = low - 1;
					while i < en {
						*h.at_mut(i, i) -= x;
						i += 1;
					}
					s = h.at(en - 1, na - 1).abs()
						+ h.at(na - 1, en - 3).abs();
					x = 0.75 * s;
					y = x;
					w = -0.4375 * s * s;
				}
				its += 1;
				itn -= 1;

				m = en - 2;
				while m >= l {
					z = *h.at(m - 1, m - 1);
					r = x - z;
					s = y - z;
					p = (r * s - w) / *h.at(m, m - 1)
						+ *h.at(m - 1, m);
					q = *h.at(m, m) - z - r - s;
					r = *h.at(m + 1, m);
					s = p.abs() + q.abs() + r.abs();
					p /= s;
					q /= s;
					r /= s;
					if m == l {
						break;
					}
					tst1 = p.abs()
						* (h.at(m - 2, m - 2).abs()
							+ z.abs()
							+ h.at(m, m).abs());
					tst2 = tst1 + h.at(m - 1, m - 2).abs()
						* (q.abs() + r.abs());
					if tst2 == tst1 {
						break;
					}
					m -= 1;
				}

				i = m + 2;
				while i <= en {
					*h.at_mut(i - 1, i - 3) = 0.0;
					if i != m + 2 {
						*h.at_mut(i - 1, i - 4) = 0.0;
					}
					i += 1;
				}
				k = m;
				while k <= na {
					not_last = k != na;
					if k != m {
						p = *h.at(k - 1, k - 2);
						q = *h.at(k, k - 2);
						r = 0.0;
						if not_last {
							r = *h.at(k + 1, k - 2);
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
								+ q * q
								+ r * r)
								.sqrt();
						} else {
							s = (p * p
								+ q * q
								+ r * r)
								.sqrt();
						}
						if k != m {
							*h.at_mut(
								k - 1,
								k - 2,
							) = -s * x;
						} else if l != m {
							*h.at_mut(
								k - 1,
								k - 2,
							) = -*h.at(
								k - 1,
								k - 2,
							);
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
								p = *h.at(
									k - 1,
									j,
								) + q * *h
									.at(
									k, j,
								);
								*h.at_mut(
									k - 1,
									j,
								) -= p * x;
								*h.at_mut(
									k, j,
								) -= p * y;
								j += 1;
							}
							j = if en < k + 3 {
								en
							} else {
								k + 3
							};
							i = 0;
							while i < j {
								p = x * *h.at(
									i,
									k - 1,
								) + y * *h
									.at(
									i, k,
								);
								*h.at_mut(
									i,
									k - 1,
								) -= p;
								*h.at_mut(
									i, k,
								) -= p * q;
								i += 1;
							}
							i = low - 1;
							while i < hgh {
								p = x * *zz.at(i, k - 1) + y * *zz.at(i, k);
								*zz.at_mut(
									i,
									k - 1,
								) -= p;
								*zz.at_mut(
									i, k,
								) -= p * q;
								i += 1;
							}
						} else {
							j = k - 1;
							while j < n {
								p = *h.at(
									k - 1,
									j,
								) + q * *h
									.at(
									k, j,
								) + r * *h
									.at(
									k + 1,
									j,
								);
								*h.at_mut(
									k - 1,
									j,
								) -= p * x;
								*h.at_mut(
									k, j,
								) -= p * y;
								*h.at_mut(
									k + 1,
									j,
								) -= p * z;
								j += 1;
							}
							j = if en < k + 3 {
								en
							} else {
								k + 3
							};
							i = 0;
							while i < j {
								p = x * *h.at(
									i,
									k - 1,
								) + y * *h
									.at(
									i, k,
								) + z * *h
									.at(
									i,
									k + 1,
								);
								*h.at_mut(
									i,
									k - 1,
								) -= p;
								*h.at_mut(
									i, k,
								) -= p * q;
								*h.at_mut(
									i,
									k + 1,
								) -= p * r;
								i += 1;
							}
							i = low - 1;
							while i < hgh {
								p = x * *zz.at(i, k - 1)
                                    + y * *zz.at(i, k)
                                    + z * *zz.at(i, k + 1);
								*zz.at_mut(
									i,
									k - 1,
								) -= p;
								*zz.at_mut(
									i, k,
								) -= p * q;
								*zz.at_mut(
									i,
									k + 1,
								) -= p * r;
								i += 1;
							}
						}
					}
					k += 1;
				}
			}

			if l == en {
				*h.at_mut(en - 1, en - 1) = x + t;
				wr[en - 1] = *h.at(en - 1, en - 1);
				wi[en - 1] = 0.0;
				en = na;
				continue 'outer;
			}

			y = *h.at(na - 1, na - 1);
			w = *h.at(en - 1, na - 1) * *h.at(na - 1, en - 1);
			p = (y - x) / 2.0;
			q = p * p + w;
			z = q.abs().sqrt();
			*h.at_mut(en - 1, en - 1) = x + t;
			x = *h.at(en - 1, en - 1);
			*h.at_mut(na - 1, na - 1) = y + t;
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
				x = *h.at(en - 1, na - 1);
				s = x.abs() + z.abs();
				p = x / s;
				q = z / s;
				r = (p * p + q * q).sqrt();
				p /= r;
				q /= r;
				j = na - 1;
				while j < n {
					z = *h.at(na - 1, j);
					*h.at_mut(na - 1, j) =
						q * z + p * *h.at(en - 1, j);
					*h.at_mut(en - 1, j) =
						q * *h.at(en - 1, j) - p * z;
					j += 1;
				}
				i = 0;
				while i < en {
					z = *h.at(i, na - 1);
					*h.at_mut(i, na - 1) =
						q * z + p * *h.at(i, en - 1);
					*h.at_mut(i, en - 1) =
						q * *h.at(i, en - 1) - p * z;
					i += 1;
				}
				i = low - 1;
				while i < hgh {
					z = *zz.at(i, na - 1);
					*zz.at_mut(i, na - 1) =
						q * z + p * *zz.at(i, en - 1);
					*zz.at_mut(i, en - 1) =
						q * *zz.at(i, en - 1) - p * z;
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
					*h.at_mut(en - 1, en - 1) = 1.0;
					if na != 0 {
						i = en - 2;
						loop {
							w = *h.at(i, i) - p;
							r = 0.0;
							j = m - 1;
							while j < en {
								r += *h.at(
									i, j,
								) * *h.at(
									j,
									en - 1,
								);
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
									*h.at_mut(i, en - 1) = -(r / t);
								} else {
									x = *h.at(i, i + 1);
									y = *h.at(i + 1, i);
									q = (wr[i] - p) * (wr[i] - p)
                                        + wi[i] * wi[i];
									t = (x * s - z * r) / q;
									*h.at_mut(i, en - 1) = t;
									if x.abs() > z.abs() {
                                        *h.at_mut(i + 1, en - 1) = (-r - w * t) / x;
                                    } else {
                                        *h.at_mut(i + 1, en - 1) = (-s - y * t) / z;
                                    }
								}
								t = h.at(
									i,
									en - 1,
								)
								.abs();
								if t != 0.0 {
									tst1 = t;
									tst2 = tst1 + 1.0 / tst1;
									if tst2 <= tst1 {
                                        j = i;
                                        while j < en {
                                            *h.at_mut(j, en - 1) /= t;
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
					if h.at(en - 1, na - 1).abs()
						> h.at(na - 1, en - 1).abs()
					{
						*h.at_mut(na - 1, na - 1) = q
							/ *h.at(en - 1, na - 1);
						*h.at_mut(na - 1, en - 1) = (p
							- *h.at(
								en - 1,
								en - 1,
							))
							/ *h.at(en - 1, na - 1);
					} else {
						let (cr, ci) = Self::mcdiv(
							0.0,
							-*h.at(na - 1, en - 1),
							*h.at(na - 1, na - 1)
								- p,
							q,
						);
						*h.at_mut(na - 1, na - 1) = cr;
						*h.at_mut(na - 1, en - 1) = ci;
					}
					*h.at_mut(en - 1, na - 1) = 0.0;
					*h.at_mut(en - 1, en - 1) = 1.0;
					if en != 2 {
						i = en - 3;
						loop {
							w = *h.at(i, i) - p;
							ra = 0.0;
							sa = 0.0;
							j = m - 1;
							while j < en {
								ra += *h.at(
									i, j,
								) * *h
									.at(
									j,
									na - 1,
								);
								sa += *h.at(
									i, j,
								) * *h
									.at(
									j,
									en - 1,
								);
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
									*h.at_mut(i, na - 1) = cr;
									*h.at_mut(i, en - 1) = ci;
								} else {
									x = *h.at(i, i + 1);
									y = *h.at(i + 1, i);
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
									*h.at_mut(i, na - 1) = cr;
									*h.at_mut(i, en - 1) = ci;
									if x.abs() > z.abs() + q.abs() {
                                        *h.at_mut(i + 1, na - 1) = (q * *h.at(i, en - 1)
                                            - w * *h.at(i, na - 1)
                                            - ra)
                                            / x;
                                        *h.at_mut(i + 1, en - 1) = (-sa
                                            - w * *h.at(i, en - 1)
                                            - q * *h.at(i, na - 1))
                                            / x;
                                    } else {
                                        let (cr, ci) = Self::mcdiv(
                                            -r - y * *h.at(i, na - 1),
                                            -s - y * *h.at(i, en - 1),
                                            z,
                                            q,
                                        );
                                        *h.at_mut(i + 1, na - 1) = cr;
                                        *h.at_mut(i + 1, en - 1) = ci;
                                    }
								}
								t = if h.at(i, na - 1).abs() > h.at(i, en - 1).abs() {
                                    h.at(i, na - 1).abs()
                                } else {
                                    h.at(i, en - 1).abs()
                                };
								if t != 0.0 {
									tst1 = t;
									tst2 = tst1 + 1.0 / tst1;
									if tst2 <= tst1 {
                                        j = i;
                                        while j < en {
                                            *h.at_mut(j, na - 1) /= t;
                                            *h.at_mut(j, en - 1) /= t;
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
						*zz.at_mut(i, j) = *h.at(i, j);
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
						z += *zz.at(i, k) * *h.at(k, j);
						k += 1;
					}
					*zz.at_mut(i, j) = z;
					i += 1;
				}
			}
		}

		Ok(())
	}

	fn eltran(
		a: &RowMatrix<f64>,
		zz: &mut RowMatrix<f64>,
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
				*zz.at_mut(i, j) = 0.0;
				*zz.at_mut(j, i) = 0.0;
				j += 1;
			}
			*zz.at_mut(i, i) = 1.0;
			i += 1;
		}
		if n <= 2 {
			return;
		}
		m = n - 1;
		while m >= 2 {
			i = m;
			while i < n {
				*zz.at_mut(i, m - 1) = *a.at(i, m - 2);
				i += 1;
			}
			i = ordr[m - 1];
			if i != m {
				j = m - 1;
				while j < n {
					*zz.at_mut(m - 1, j) = *zz.at(i - 1, j);
					*zz.at_mut(i - 1, j) = 0.0;
					j += 1;
				}
				*zz.at_mut(i - 1, m - 1) = 1.0;
			}
			m -= 1;
		}
	}

	fn luinverse(
		inmat: &RowMatrix<f64>,
		imtrx: &mut RowMatrix<f64>,
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
		let mut omtrx = inmat.clone();
		let mut wk = vec![0.0; size];

		aw = 1.0;
		i = 0;
		while i < size {
			maxb = 0.0;
			j = 0;
			while j < size {
				if omtrx.at(i, j).abs() > maxb {
					maxb = omtrx.at(i, j).abs();
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
				sum = *omtrx.at(i, j);
				k = 0;
				while k < i {
					sum -= *omtrx.at(i, k)
						* *omtrx.at(k, j);
					k += 1;
				}
				*omtrx.at_mut(i, j) = sum;
				i += 1;
			}
			maxb = 0.0;
			i = j;
			while i < size {
				sum = *omtrx.at(i, j);
				k = 0;
				while k < j {
					sum -= *omtrx.at(i, k)
						* *omtrx.at(k, j);
					k += 1;
				}
				*omtrx.at_mut(i, j) = sum;
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
					tmp = *omtrx.at(maxi, k);
					*omtrx.at_mut(maxi, k) =
						*omtrx.at(j, k);
					*omtrx.at_mut(j, k) = tmp;
					k += 1;
				}
				aw = -aw;
				wk[maxi] = wk[j];
			}
			index[j] = maxi;
			if *omtrx.at(j, j) == 0.0 {
				*omtrx.at_mut(j, j) = MACHINE_EPSILON;
			}
			if j != size - 1 {
				tmp = 1.0 / *omtrx.at(j, j);
				i = j + 1;
				while i < size {
					*omtrx.at_mut(i, j) *= tmp;
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
						sum -= *omtrx.at(i, j) * wk[j];
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
					sum -= *omtrx.at(i, j) * wk[j];
					j += 1;
				}
				wk[i] = sum / *omtrx.at(i, i);
				if i == 0 {
					break;
				}
				i -= 1;
			}
			ix = 0;
			while ix < size {
				*imtrx.at_mut(ix, jx) = wk[ix];
				ix += 1;
			}
			jx += 1;
		}

		Ok(())
	}
}
