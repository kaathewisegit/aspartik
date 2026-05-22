use anyhow::Result;
use fork_union::{SyncMutPtr, ThreadPool};
use math::Probability;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::deduplicate;
use crate::{
	Transitions,
	calculator::{Calculator, CalculatorConfig},
	clock::PyClock,
	parameters::{Parameter, PyReal, PyTree, Real},
	substitution::PySubstitution4,
	substitution::SubstitutionModel,
};
use data::PyMsa;
use sk::SkBuf;
use stats::distribution::{ContinuousCDF, Gamma};
use util::atomic::{MonotonicBool, MonotonicF64};

#[pyclass(module = "aspartik.b3.likelihoods", frozen)]
pub struct GammaLikelihood {
	calculators: Mutex<Vec<Box<dyn Calculator<4, f64> + Send + Sync>>>,

	substitution: Mutex<Box<dyn SubstitutionModel<4, f64> + Send>>,
	clock: Py<PyClock>,
	alpha: Py<PyReal>,
	transitions: Mutex<Vec<Transitions<4, f64>>>,
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
		substitution: PySubstitution4,
		num_categories: usize,
		alpha: Py<PyReal>,
		clock: Py<PyClock>,
		calculator: CalculatorConfig,
	) -> Result<Self> {
		let (samples, weights) = deduplicate(msa.get());

		let mut calculators = Vec::new();
		let mut transitions = Vec::new();

		let num_nodes = tree.get().num_nodes();

		for _ in 0..num_categories {
			calculators.push(calculator
				.make4(samples.clone(), weights.clone())?);

			let transition = Transitions::new(num_nodes);
			transitions.push(transition);
		}

		let mut categories = SkBuf::repeat(0.0, num_categories);
		update_categories(&alpha.get().inner(), &mut categories);

		let pool = ThreadPool::try_spawn(categories.len())?;

		let out = Self {
			calculators: Mutex::new(calculators),

			clock,
			substitution: Mutex::new(Box::new(substitution)),
			alpha,
			transitions: Mutex::new(transitions),
			tree,

			categories: Mutex::new(categories),

			pool: Mutex::new(pool),

			cache: f64::NAN.into(),
			last: f64::NAN.into(),
			launched_update: false.into(),
		};
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

		clock.update()?;
		clock.mark_tree(&mut tree);

		if substitution.update()? {
			tree.mark_all_edges_updated();
		}

		let is_changed = tree.is_changed() || alpha.is_changed();

		if !is_changed {
			self.last.store(self.cache.load());
			return Ok(self.cache.load());
		}

		let clock_rate = clock.get_rate();
		let mut categories = self.categories.lock();

		// recalculate categories if alpha has changed
		if alpha.is_changed() {
			update_categories(&alpha, &mut categories);
		}

		for (transition, category) in
			transitions.iter_mut().zip(categories.iter())
		{
			transition.update(
				&mut tree,
				clock_rate * category,
				&**substitution,
			)?;
		}

		let frequencies = substitution.get_frequencies();
		drop(clock);
		drop(substitution);
		drop(tree);

		self.launched_update.store(true);

		let likelihood = Mutex::new(0.0);

		let calculators = &mut *self.calculators.lock();
		let calc_ptr = SyncMutPtr::new(calculators.as_mut_ptr());

		self.pool.lock().for_n(categories.len(), |prong| {
			let i = prong.task_index;
			let transition = &transitions[i];
			let tree = self.tree.get().inner();

			// SAFETY: `i` is less thant `calculator.len()`, all
			// accesses will be disjoint due to `for_n`.
			let calculator = unsafe { &mut *calc_ptr.get(i) };

			let out = calculator
				.likelihood(tree, transition, frequencies)
				.unwrap();
			*likelihood.lock() += out;
		});

		let likelihood = *likelihood.lock();

		self.last.store(likelihood);
		Ok(likelihood)
	}

	pub fn accept(&self) -> Result<()> {
		self.cache.store(self.last.load());
		if self.launched_update.load() {
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
	let dist = Gamma::new(alpha, 1.0 / alpha).unwrap();
	let mut sum = 0.0;
	let num_categories = categories.len();

	for i in 0..num_categories {
		let p = (2.0 * i as f64 + 1.0) / (2.0 * num_categories as f64);
		let modifier = dist.inverse_cdf(Probability::new(p));
		sum += modifier;

		categories.set(i, modifier);
	}

	let mean = sum / num_categories as f64;

	for i in 0..num_categories {
		let modifier = categories[i];
		categories.set(i, modifier / mean);
	}
}
