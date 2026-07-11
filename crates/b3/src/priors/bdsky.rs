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
		})
	}
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct BirthDeathSkyline {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	origin: Py<PyReal>,
	#[pyo3(get)]
	become_uninfection_rate: Py<PyReal>,
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

fn log_q(a: f64, c: f64, ti: f64, t: f64) -> f64 {
	let diff = t - ti;
	let ex = (a * diff).exp();
	let denom = ex * (1.0 - c) + (1.0 + c);
	(4.0_f64).ln() + a * diff - 2.0 * denom.ln()
}

fn lineage_count_at_time(height: f64, tree: &Tree) -> i64 {
	let mut count = 1;
	for internal in tree.internals() {
		if tree.height_of(*internal) > height {
			count += 1;
		}
	}
	for leaf in tree.leaves() {
		if tree.height_of(*leaf) >= height {
			count -= 1;
		}
	}
	count
}

#[pymethods]
impl BirthDeathSkyline {
	#[new]
	#[pyo3(signature = (
		tree,
		origin,
		become_uninfection_rate,
		reproductive_number,
		sampling_proportion,
	))]
	fn new(
		tree: Py<PyTree>,
		origin: Py<PyReal>,
		become_uninfection_rate: Py<PyReal>,
		reproductive_number: Py<PyRealVector>,
		sampling_proportion: Py<PyReal>,
	) -> Result<Self> {
		let size = reproductive_number.get().inner().len();
		ensure!(size >= 2, "Expected at least two skyline intervals");

		Ok(Self {
			tree,
			origin,
			become_uninfection_rate,
			reproductive_number,
			sampling_proportion,
			scratch: Scratch::new(size),
		})
	}

	fn probability(&self) -> Result<f64> {
		let s = &mut *self.scratch.lock();
		let tree = self.tree.get().inner();
		let origin = self.origin.get().inner().value();
		let become_uninfection_rate =
			self.become_uninfection_rate.get().inner().value();
		let reproductive_number =
			self.reproductive_number.get().inner();
		let sampling_proportion =
			self.sampling_proportion.get().inner().value();

		let n = reproductive_number.len();
		let root_height = tree.height_of(*tree.root());

		if root_height >= origin {
			return Ok(f64::NEG_INFINITY);
		}

		let iw = origin / n as f64; // equidistant interval width

		for i in 0..n {
			s.birth[i] = reproductive_number[i]
				* become_uninfection_rate;
		}
		for i in 0..n {
			s.psi[i] =
				sampling_proportion * become_uninfection_rate;
		}
		for i in 0..n {
			s.death[i] = become_uninfection_rate - s.psi[i];
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
			let ti = (i + 2) as f64 * iw; // times[i + 1]
			let t = (i + 1) as f64 * iw; // times[i]
			s.p0_next[i + 1] = p0(
				s.birth[i + 1],
				s.death[i + 1],
				s.psi[i + 1],
				s.ai[i + 1],
				s.bi[i + 1],
				ti,
				t,
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

		let index_of = |x: f64| -> usize {
			((x / iw).max(0.0) as usize).min(n - 1)
		};

		let mut log_p = 0.0;

		let p0_origin = p0(
			s.birth[0], s.death[0], s.psi[0], s.ai[0], s.bi[0], iw,
			0.0,
		);
		if p0_origin == 1.0 {
			return Ok(f64::NEG_INFINITY);
		}
		log_p += log_q(s.ai[0], s.bi[0], iw, 0.0)
			- (1.0 - p0_origin).ln();

		for internal in tree.internals() {
			let x = origin - tree.height_of(*internal);
			let idx = index_of(x);
			let ti = (idx + 1) as f64 * iw;
			log_p += s.birth[idx].ln()
				+ log_q(s.ai[idx], s.bi[idx], ti, x);
		}

		for leaf in tree.leaves() {
			let y = origin - tree.height_of(*leaf);
			let idx = index_of(y);
			if s.psi[idx] == 0.0 {
				return Ok(f64::NEG_INFINITY);
			}
			let ti = (idx + 1) as f64 * iw;
			log_p += s.psi[idx].ln()
				- log_q(s.ai[idx], s.bi[idx], ti, y);
		}

		for j in 1..n {
			let time = j as f64 * iw; // times[j − 1]
			let lineage_height = origin - time;
			let n_lineages =
				lineage_count_at_time(lineage_height, &tree);
			if n_lineages > 0 {
				let ti = (j + 1) as f64 * iw; // times[j]
				log_p += n_lineages as f64
					* log_q(s.ai[j], s.bi[j], ti, time);
			}
		}

		Ok(log_p)
	}

	fn is_changed(&self) -> bool {
		self.tree.get().is_changed()
			|| self.origin.get().is_changed()
			|| self.become_uninfection_rate.get().is_changed()
			|| self.reproductive_number.get().is_changed()
			|| self.sampling_proportion.get().is_changed()
	}
}
