use anyhow::{Result, ensure};
use parking_lot::Mutex;
use pyo3::prelude::*;

use crate::parameters::{PyReal, PyRealVector, PyTree, Tree};

#[derive(Debug)]
struct Scratch {
	birth: Vec<f64>,
	death: Vec<f64>,
	psi: Vec<f64>,
	ai: Vec<f64>,
	bi: Vec<f64>,
	p0_next: Vec<f64>,
	times: Vec<f64>,
	lineage_heights: Vec<f64>,
	lineage_counts: Vec<i64>,
}

impl Scratch {
	fn new(n: usize) -> Mutex<Self> {
		Mutex::new(Self {
			birth: vec![0.0; n],
			death: vec![0.0; n],
			psi: vec![0.0; n],
			ai: vec![0.0; n],
			bi: vec![0.0; n],
			p0_next: vec![0.0; n],
			times: vec![0.0; n + 1],
			lineage_heights: vec![0.0; n],
			lineage_counts: vec![0; n],
		})
	}
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct BirthDeathSkyline {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	origin: Option<Py<PyReal>>,
	#[pyo3(get)]
	interval_times: Option<Py<PyReal>>,
	#[pyo3(get)]
	become_uninfectious_rate: Py<PyReal>,
	#[pyo3(get)]
	reproductive_number: Py<PyRealVector>,
	#[pyo3(get)]
	sampling_proportion: Py<PyReal>,
	scratch: Mutex<Scratch>,
}

fn ai(b: f64, g: f64, psi: f64) -> f64 {
	let diff = b - g - psi;
	(diff * diff + 4.0 * b * psi).sqrt()
}

fn bi(b: f64, g: f64, psi: f64, a: f64, p0: f64) -> f64 {
	((1.0 - 2.0 * p0) * b + g + psi) / a
}

/// ti is the interval boundry
fn p0(b: f64, g: f64, psi: f64, a: f64, c: f64, ti: f64, t: f64) -> f64 {
	let ex = (a * (t - ti)).exp();
	let numer = (1.0 + c) - (1.0 - c) * ex;
	let denom = (1.0 + c) + ex * (1.0 - c);
	(b + g + psi - a * numer / denom) / (2.0 * b)
}

fn q(a: f64, c: f64, ti: f64, t: f64) -> f64 {
	let diff = t - ti;
	let ex = (a * diff).exp();
	let denom = ex * (1.0 - c) + (1.0 + c);
	4.0 * ex / (denom * denom)
}

fn mul_scale(probability: &mut f64, scale: &mut i32, mul: f64) {
	const THRESHOLD: f64 = 0.000000002061153622438558; // e^-20
	const SCALE: f64 = 485165195.4097903; // e^20

	*probability *= mul;
	assert_ne!(*probability, 0.0);

	while *probability < THRESHOLD {
		*probability *= SCALE;
		*scale -= 20;
	}
	while *probability > 1.0 {
		*probability *= THRESHOLD;
		*scale += 20;
	}
}

fn update_lineage_counts(heights: &[f64], tree: &Tree, counts: &mut [i64]) {
	counts.fill(1);
	for internal in tree.internals() {
		let h = tree.height_of(*internal);
		for (&height, count) in heights.iter().zip(counts.iter_mut()) {
			*count += i64::from(h > height);
		}
	}
	for leaf in tree.leaves() {
		let h = tree.height_of(*leaf);
		for (&height, count) in heights.iter().zip(counts.iter_mut()) {
			*count -= i64::from(h > height);
		}
	}
}

fn index_of(times: &[f64], x: f64, n: usize) -> usize {
	times.iter().rposition(|&t| t <= x).unwrap_or(0).min(n - 1)
}

fn update_change_times(
	times: &mut [f64],
	origin: Option<f64>,
	interval: Option<f64>,
	root_height: f64,
) {
	let n = times.len() - 1;
	let max_time = origin.unwrap_or(root_height);
	let interval_width = interval.unwrap_or(max_time / n as f64);

	times[0] = 0.0;
	#[expect(clippy::needless_range_loop)]
	for i in 1..n {
		times[i] = interval_width * i as f64;
	}
	times[n] = max_time;
}

#[pymethods]
impl BirthDeathSkyline {
	#[new]
	#[pyo3(signature = (
		tree,
		become_uninfectious_rate,
		reproductive_number,
		sampling_proportion,
		*,
		origin=None,
		interval_times=None,
	))]
	fn new(
		tree: Py<PyTree>,
		become_uninfectious_rate: Py<PyReal>,
		reproductive_number: Py<PyRealVector>,
		sampling_proportion: Py<PyReal>,
		origin: Option<Py<PyReal>>,
		interval_times: Option<Py<PyReal>>,
	) -> Result<Self> {
		let size = reproductive_number.get().inner().len();
		ensure!(size >= 2, "Expected at least two skyline intervals");

		Ok(Self {
			tree,
			origin,
			interval_times,
			become_uninfectious_rate,
			reproductive_number,
			sampling_proportion,
			scratch: Scratch::new(size),
		})
	}

	fn probability(&self) -> Result<f64> {
		let s = &mut *self.scratch.lock();
		let tree = self.tree.get().inner();
		let become_uninfectious_rate =
			self.become_uninfectious_rate.get().inner().value();
		let reproductive_number =
			self.reproductive_number.get().inner();
		let sampling_proportion =
			self.sampling_proportion.get().inner().value();
		let origin =
			self.origin.as_ref().map(|o| o.get().inner().value());
		let interval_times = self
			.interval_times
			.as_ref()
			.map(|it| it.get().inner().value());

		let root_height = tree.height_of(*tree.root());

		if origin.is_some_and(|o| o < root_height) {
			return Ok(f64::NEG_INFINITY);
		}

		let n = reproductive_number.len();

		update_change_times(
			&mut s.times,
			origin,
			interval_times,
			root_height,
		);

		if s.times[n] < s.times[n - 1] {
			return Ok(f64::NEG_INFINITY);
		}

		for i in 0..n {
			s.birth[i] = reproductive_number[i]
				* become_uninfectious_rate;
		}
		for i in 0..n {
			s.psi[i] =
				sampling_proportion * become_uninfectious_rate;
		}
		for i in 0..n {
			s.death[i] = become_uninfectious_rate - s.psi[i];
		}

		for i in 0..n {
			s.ai[i] = ai(s.birth[i], s.death[i], s.psi[i]);
		}

		s.bi[n - 1] = bi(
			s.birth[n - 1],
			s.death[n - 1],
			s.psi[n - 1],
			s.ai[n - 1],
			1.0,
		);

		for i in (0..n - 1).rev() {
			s.p0_next[i + 1] = p0(
				s.birth[i + 1],
				s.death[i + 1],
				s.psi[i + 1],
				s.ai[i + 1],
				s.bi[i + 1],
				s.times[i + 2],
				s.times[i + 1],
			);
			if (s.p0_next[i + 1] - 1.0).abs() < 1e-10 {
				return Ok(f64::NEG_INFINITY);
			}
			s.bi[i] = bi(
				s.birth[i],
				s.death[i],
				s.psi[i],
				s.ai[i],
				s.p0_next[i + 1],
			);
		}

		let mut out = 1.0;
		let mut scale = 0;

		let p0_origin = p0(
			s.birth[0], s.death[0], s.psi[0], s.ai[0], s.bi[0],
			s.times[1], s.times[0],
		);
		if p0_origin == 1.0 {
			return Ok(f64::NEG_INFINITY);
		}
		let q0 = q(s.ai[0], s.bi[0], s.times[1], s.times[0]);
		mul_scale(&mut out, &mut scale, q0 / (1.0 - p0_origin));

		let last = s.times.last().unwrap();

		for internal in tree.internals() {
			let x = last - tree.height_of(*internal);
			let idx = index_of(&s.times, x, n);
			let qi = q(s.ai[idx], s.bi[idx], s.times[idx + 1], x);
			mul_scale(&mut out, &mut scale, s.birth[idx] * qi);
		}

		for leaf in tree.leaves() {
			let y = last - tree.height_of(*leaf);
			let idx = index_of(&s.times, y, n);
			if s.psi[idx] == 0.0 {
				return Ok(f64::NEG_INFINITY);
			}
			let qi = q(s.ai[idx], s.bi[idx], s.times[idx + 1], y);
			mul_scale(&mut out, &mut scale, s.psi[idx] / qi);
		}

		for j in 1..n {
			s.lineage_heights[j] = last - s.times[j];
		}
		update_lineage_counts(
			&s.lineage_heights[1..n],
			&tree,
			&mut s.lineage_counts[1..n],
		);

		for j in 1..n {
			let n_lineages = s.lineage_counts[j];
			if n_lineages > 0 {
				let qj = q(
					s.ai[j],
					s.bi[j],
					s.times[j + 1],
					s.times[j],
				);
				for _ in 0..n_lineages {
					mul_scale(&mut out, &mut scale, qj);
				}
			}
		}

		Ok(out.ln() + f64::from(scale))
	}

	fn is_changed(&self) -> bool {
		self.tree.get().is_changed()
			|| self.become_uninfectious_rate.get().is_changed()
			|| self.reproductive_number.get().is_changed()
			|| self.sampling_proportion.get().is_changed()
			|| self.origin
				.as_ref()
				.is_some_and(|it| it.get().is_changed())
			|| self.interval_times
				.as_ref()
				.is_some_and(|it| it.get().is_changed())
	}
}
