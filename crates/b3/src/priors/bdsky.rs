use anyhow::{Result, ensure};
use parking_lot::Mutex;
use pyo3::prelude::*;

use crate::parameters::{PyReal, PyRealVector, PyTree, RealVector, Tree};

#[derive(Debug, Default)]
struct Calculation {
	interval_ends: Vec<f64>,
	a: Vec<f64>,
	b: Vec<f64>,
	origin_age: f64,
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.priors", frozen)]
pub struct BirthDeathSkyline {
	#[pyo3(get)]
	tree: Py<PyTree>,
	#[pyo3(get)]
	times: Py<PyRealVector>,
	#[pyo3(get)]
	birth_rates: Py<PyRealVector>,
	#[pyo3(get)]
	death_rates: Py<PyRealVector>,
	#[pyo3(get)]
	sampling_rates: Py<PyRealVector>,
	#[pyo3(get)]
	origin: Py<PyReal>,
	#[pyo3(get)]
	relative_death: bool,
	#[pyo3(get)]
	times_start_from_origin: bool,
	#[pyo3(get)]
	condition_on_survival: bool,
	scratch: Mutex<Calculation>,
}

#[pymethods]
impl BirthDeathSkyline {
	#[new]
	#[pyo3(signature = (
		tree,
		times,
		birth_rates,
		death_rates,
		sampling_rates,
		origin,
		relative_death = false,
		times_start_from_origin = true,
		condition_on_survival = true,
	))]
	#[expect(clippy::too_many_arguments)]
	fn new(
		tree: Py<PyTree>,
		times: Py<PyRealVector>,
		birth_rates: Py<PyRealVector>,
		death_rates: Py<PyRealVector>,
		sampling_rates: Py<PyRealVector>,
		origin: Py<PyReal>,
		relative_death: bool,
		times_start_from_origin: bool,
		condition_on_survival: bool,
	) -> Result<Self> {
		let size = times.get().inner().len();
		ensure!(size >= 2, "Expected at least two skyline intervals");
		ensure_rate_len(
			"birth_rates",
			birth_rates.get().inner().len(),
			size,
		)?;
		ensure_rate_len(
			"death_rates",
			death_rates.get().inner().len(),
			size,
		)?;
		ensure_rate_len(
			"sampling_rates",
			sampling_rates.get().inner().len(),
			size,
		)?;

		Ok(Self {
			tree,
			times,
			birth_rates,
			death_rates,
			sampling_rates,
			origin,
			relative_death,
			times_start_from_origin,
			condition_on_survival,
			scratch: Mutex::new(Calculation::default()),
		})
	}

	fn probability(&self) -> Result<f64> {
		let tree = self.tree.get().inner();
		let times = self.times.get();
		let times = times.inner();
		let birth_rates = self.birth_rates.get();
		let birth_rates = birth_rates.inner();
		let death_rates = self.death_rates.get();
		let death_rates = death_rates.inner();
		let sampling_rates = self.sampling_rates.get();
		let sampling_rates = sampling_rates.inner();
		let origin = self.origin.get().inner().value();

		let rates = Rates {
			birth: &birth_rates,
			death: &death_rates,
			sampling: &sampling_rates,
			relative_death: self.relative_death,
		};
		let calculation = &mut *self.scratch.lock();
		pre_calculate(
			&tree,
			&times,
			&rates,
			origin,
			self.times_start_from_origin,
			calculation,
		)?;

		Ok(calculate_tree_log_likelihood(
			&tree,
			&birth_rates,
			&death_rates,
			&sampling_rates,
			self.relative_death,
			self.condition_on_survival,
			calculation,
		))
	}

	fn is_changed(&self) -> bool {
		self.tree.get().is_changed()
			|| self.times.get().is_changed()
			|| self.birth_rates.get().is_changed()
			|| self.death_rates.get().is_changed()
			|| self.sampling_rates.get().is_changed()
			|| self.origin.get().is_changed()
	}
}

fn ensure_rate_len(name: &str, len: usize, size: usize) -> Result<()> {
	ensure!(
		len == size,
		"{name} length should be equal to the number of skyline intervals ({size})"
	);
	Ok(())
}

fn get_birth(birth_rates: &RealVector, index: usize) -> f64 {
	birth_rates[index]
}

fn get_death(
	birth_rates: &RealVector,
	death_rates: &RealVector,
	index: usize,
	relative_death: bool,
) -> f64 {
	let death = death_rates[index];
	if relative_death {
		death * get_birth(birth_rates, index)
	} else {
		death
	}
}

fn get_sampling(sampling_rates: &RealVector, index: usize) -> f64 {
	sampling_rates[index]
}

fn a_i(birth: f64, death: f64, sampling: f64) -> f64 {
	((birth - death - sampling).powi(2) + 4.0 * birth * sampling).sqrt()
}

fn b_i(
	birth: f64,
	death: f64,
	sampling: f64,
	rho: f64,
	a: f64,
	p0: f64,
) -> f64 {
	((1.0 - 2.0 * p0 * (1.0 - rho)) * birth + death + sampling) / a
}

fn p0_at(
	birth: f64,
	death: f64,
	sampling: f64,
	a: f64,
	b: f64,
	interval_end: f64,
	time: f64,
) -> f64 {
	let exp = (a * (time - interval_end)).exp();
	(birth + death + sampling
		- a * ((1.0 + b) - (1.0 - b) * exp)
			/ ((1.0 + b) + exp * (1.0 - b)))
		/ (2.0 * birth)
}

fn p0(
	calculation: &Calculation,
	rates: &Rates<'_>,
	index: usize,
	time: f64,
) -> f64 {
	p0_at(
		get_birth(rates.birth, index),
		get_death(
			rates.birth,
			rates.death,
			index,
			rates.relative_death,
		),
		get_sampling(rates.sampling, index),
		calculation.a[index],
		calculation.b[index],
		calculation.interval_ends[index],
		time,
	)
}

fn log_q(calculation: &Calculation, index: usize, time: f64) -> f64 {
	let a = calculation.a[index];
	let b = calculation.b[index];
	let exp = (a * (time - calculation.interval_ends[index])).exp();
	4.0_f64.ln() + a * (time - calculation.interval_ends[index])
		- 2.0 * (exp * (1.0 - b) + (1.0 + b)).ln()
}

fn index(interval_ends: &[f64], time: f64) -> usize {
	if time >= *interval_ends.last().unwrap() {
		return interval_ends.len() - 1;
	}

	match interval_ends
		.binary_search_by(|probe| probe.partial_cmp(&time).unwrap())
	{
		Ok(epoch) | Err(epoch) => epoch,
	}
}

fn lineage_count_at_time(tree: &Tree, age: f64) -> usize {
	let mut count = 1usize;
	for node in tree.internals() {
		if tree.height_of(*node) > age {
			count += 1;
		}
	}
	for node in tree.leaves() {
		if tree.height_of(*node) >= age {
			count -= 1;
		}
	}
	count
}

fn update_interval_ends(
	out: &mut Vec<f64>,
	times: &RealVector,
	origin_age: f64,
	times_start_from_origin: bool,
) {
	out.clear();
	if times_start_from_origin {
		out.extend(times.iter().copied().filter(|&time| time > 0.0));
	} else {
		out.extend(times.iter().copied().filter_map(|age| {
			let time = origin_age - age;
			(time > 0.0).then_some(time)
		}));
	}

	out.sort_by(|a, b| a.partial_cmp(b).unwrap());
	out.retain(|&time| time < origin_age);
	out.push(origin_age);
}

struct Rates<'a> {
	birth: &'a RealVector,
	death: &'a RealVector,
	sampling: &'a RealVector,
	relative_death: bool,
}

fn pre_calculate(
	tree: &Tree,
	times: &RealVector,
	rates: &Rates<'_>,
	origin: f64,
	times_start_from_origin: bool,
	calculation: &mut Calculation,
) -> Result<()> {
	let root_height = tree.height_of(*tree.root());
	let origin_age = root_height + origin;
	ensure!(
		origin_age > root_height,
		"Origin must be older than the tree root"
	);

	update_interval_ends(
		&mut calculation.interval_ends,
		times,
		origin_age,
		times_start_from_origin,
	);
	let size = calculation.interval_ends.len();

	calculation.origin_age = origin_age;
	calculation.a.resize(size, 0.0);
	calculation.b.resize(size, 0.0);
	for (i, a) in calculation.a.iter_mut().enumerate() {
		*a = a_i(
			get_birth(rates.birth, i),
			get_death(
				rates.birth,
				rates.death,
				i,
				rates.relative_death,
			),
			get_sampling(rates.sampling, i),
		);
	}

	let last = size - 1;
	calculation.b[last] = b_i(
		get_birth(rates.birth, last),
		get_death(rates.birth, rates.death, last, rates.relative_death),
		get_sampling(rates.sampling, last),
		0.0,
		calculation.a[last],
		1.0,
	);

	for i in (0..last).rev() {
		let previous_p0 = p0(
			calculation,
			rates,
			i + 1,
			calculation.interval_ends[i],
		);
		ensure!(
			(previous_p0 - 1.0).abs() >= 1e-10,
			"p0 is numerically indistinguishable from 1"
		);
		calculation.b[i] = b_i(
			get_birth(rates.birth, i),
			get_death(
				rates.birth,
				rates.death,
				i,
				rates.relative_death,
			),
			get_sampling(rates.sampling, i),
			0.0,
			calculation.a[i],
			previous_p0,
		);
	}

	Ok(())
}

fn calculate_tree_log_likelihood(
	tree: &Tree,
	birth_rates: &RealVector,
	death_rates: &RealVector,
	sampling_rates: &RealVector,
	relative_death: bool,
	condition_on_survival: bool,
	calculation: &Calculation,
) -> f64 {
	let rates = Rates {
		birth: birth_rates,
		death: death_rates,
		sampling: sampling_rates,
		relative_death,
	};

	let mut log_p = log_q(calculation, 0, 0.0);
	if condition_on_survival {
		let p0_at_origin = p0(calculation, &rates, 0, 0.0);
		if p0_at_origin == 1.0 {
			return f64::NEG_INFINITY;
		}
		log_p -= (1.0 - p0_at_origin).ln();
	}

	for node in tree.internals() {
		let x = calculation.origin_age - tree.height_of(*node);
		let index = index(&calculation.interval_ends, x);
		log_p += get_birth(birth_rates, index).ln()
			+ log_q(calculation, index, x);
	}

	for node in tree.leaves() {
		let y = calculation.origin_age - tree.height_of(*node);
		let index = index(&calculation.interval_ends, y);
		let sampling = get_sampling(sampling_rates, index);
		if sampling == 0.0 {
			return f64::NEG_INFINITY;
		}
		log_p += sampling.ln() - log_q(calculation, index, y);
	}

	for j in 1..calculation.interval_ends.len() {
		let boundary_time = calculation.interval_ends[j - 1];
		let boundary_age = calculation.origin_age - boundary_time;
		let n = lineage_count_at_time(tree, boundary_age);
		if n > 0 {
			log_p +=
				n as f64 * log_q(calculation, j, boundary_time);
		}
	}

	log_p
}
