use anyhow::Result;
use fork_union::{SyncMutPtr, ThreadPool};
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::{deduplicate, log_sum_exp};
use crate::{
	Transitions,
	calculator::{Calculator, CalculatorConfig},
	clock::PyClock,
	parameters::{Parameter, PyReal, PyTree, Real},
	substitution::Substitution,
};
use data::PyMsa;
use sk::SkBuf;
use stats::distribution::{ContinuousCDF, Gamma};
use util::atomic::{MonotonicBool, MonotonicF64};

#[pyclass(module = "aspartik.b3.likelihoods", frozen)]
pub struct GammaLikelihood {
	calculators: Mutex<Vec<Box<dyn Calculator<f64> + Send + Sync>>>,
	weights: Vec<u32>,

	substitution: Mutex<Substitution>,
	clock: Py<PyClock>,
	alpha: Py<PyReal>,
	transitions: Mutex<Vec<Transitions>>,
	tree: Py<PyTree>,

	categories: Mutex<SkBuf<f64>>,

	pool: Mutex<ThreadPool>,

	cache: MonotonicF64,
	last: MonotonicF64,
	launched_update: MonotonicBool,
}

#[pymethods]
impl GammaLikelihood {
	#[new]
	fn new(
		msa: Py<PyMsa>,
		tree: Py<PyTree>,
		substitution: Substitution,
		num_categories: usize,
		alpha: Py<PyReal>,
		clock: Py<PyClock>,
		calculator: CalculatorConfig,
	) -> Result<Self> {
		let (samples, weights, _) = deduplicate(msa.get());

		let mut calculators = Vec::new();
		let mut transitions = Vec::new();

		let num_nodes = tree.get().num_nodes();

		for _ in 0..num_categories {
			calculators.push(calculator
				.make4(samples.clone(), weights.len())?);

			let transition = Transitions::new(4, num_nodes);
			transitions.push(transition);
		}

		let mut categories = SkBuf::repeat(0.0, num_categories);
		update_categories(&alpha.get().inner(), &mut categories);

		let pool = ThreadPool::try_spawn(1)?;

		let out = Self {
			calculators: Mutex::new(calculators),
			weights,

			clock,
			substitution: Mutex::new(substitution),
			alpha,
			transitions: Mutex::new(transitions),
			tree,

			categories: Mutex::new(categories),

			pool: Mutex::new(pool),

			cache: f64::NAN.into(),
			last: f64::NAN.into(),
			launched_update: false.into(),
		};
		out.tree.get().inner().mark_all_edges_updated();
		out.likelihood()?;
		// likelihood sets `last` and accept updates the cache, so
		// neither cache nor last will be NaN.
		out.accept()?;
		Ok(out)
	}

	pub fn likelihood(&self) -> Result<f64> {
		let mut tree = self.tree.get().inner();
		let mut transitions = self.transitions.lock();
		let mut clock = self.clock.get().inner();
		let mut substitution = self.substitution.lock();
		let alpha = self.alpha.get().inner();

		clock.update(&mut tree)?;

		if substitution.update()? {
			tree.mark_all_edges_updated();
		}

		let is_changed = tree.is_changed() || alpha.is_changed();

		if !is_changed {
			self.last.store(self.cache.load());
			return Ok(self.cache.load());
		}

		let mut categories = self.categories.lock();
		let num_categories = categories.len();

		// recalculate categories if alpha has changed
		if alpha.is_changed() {
			tree.mark_all_edges_updated();
			update_categories(&alpha, &mut categories);
		}

		for (transition, category) in
			transitions.iter_mut().zip(categories.iter())
		{
			transition.update(
				&mut tree,
				&mut **substitution,
				|edge| clock.get_rate(edge) * category,
			)?;
		}

		drop(clock);

		self.launched_update.store(true);

		let calculators = &mut *self.calculators.lock();
		let calc_ptr = SyncMutPtr::new(calculators.as_mut_ptr());

		self.pool.lock().for_n(num_categories, |prong| {
			let i = prong.task_index;
			let transition = &transitions[i];

			// SAFETY: `i` is less thant `calculator.len()`, all
			// accesses will be disjoint due to `for_n`.
			let calculator = unsafe { &mut *calc_ptr.get(i) };

			calculator.propose(&tree, transition).unwrap();
		});

		let num_patterns = calculators[0].num_patterns();

		let mut likelihood = 0.0;
		let div = (1.0 / (num_categories as f64)).ln();
		let mut sums = vec![0.0; num_categories];
		for i in 0..num_patterns {
			for c in 0..num_categories {
				sums[c] = calculators[c].likelihoods()[i];
			}
			let sum = log_sum_exp(&sums) + div;
			likelihood += sum * f64::from(self.weights[i]);
		}

		self.last.store(likelihood);
		Ok(likelihood)
	}

	pub fn accept(&self) -> Result<()> {
		self.cache.store(self.last.load());
		if self.launched_update.load() {
			self.categories.lock().accept();
			for calculator in self.calculators.lock().iter_mut() {
				calculator.accept()?;
			}
			for transition in self.transitions.lock().iter_mut() {
				transition.accept();
			}
			self.substitution.lock().accept();
			self.clock.get().inner().accept();
		}
		self.launched_update.store(false);
		Ok(())
	}

	pub fn reject(&self) -> Result<()> {
		if self.launched_update.load() {
			self.categories.lock().reject();
			for calculator in self.calculators.lock().iter_mut() {
				calculator.reject()?;
			}
			for transition in self.transitions.lock().iter_mut() {
				transition.reject();
			}
			self.substitution.lock().reject();
			self.clock.get().inner().reject();
		}
		self.launched_update.store(false);
		Ok(())
	}
}

fn update_categories(alpha: &Real, categories: &mut SkBuf<f64>) {
	let alpha = alpha.value();
	let dist = Gamma::new(alpha, 1.0).unwrap();
	let mut sum = 0.0;
	let num_categories = categories.len();

	for i in 0..num_categories {
		let p = (2.0 * i as f64 + 1.0) / (2.0 * num_categories as f64);
		let modifier = dist.inverse_cdf(p);
		sum += modifier;

		categories.set(i, modifier);
	}

	let mean = sum / num_categories as f64;

	for i in 0..num_categories {
		let modifier = categories[i];
		categories.set(i, modifier / mean);
	}
}
