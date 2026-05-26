use anyhow::Result;
use fork_union::{SyncMutPtr, ThreadPool};
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::deduplicate;
use crate::{
	Transitions,
	calculator::{Calculator, CalculatorConfig},
	clock::PyClock,
	parameters::{Parameter, PyClassVector, PyTree},
	substitution::Substitution,
};
use buffer::SliceBuffer;
use data::PyMsa;
use util::atomic::{MonotonicBool, MonotonicF64};

#[pyclass(module = "aspartik.b3.likelihoods", frozen)]
pub struct HeteroLikelihood {
	calculators: Mutex<Vec<Box<dyn Calculator<f64> + Send + Sync>>>,

	categories: Py<PyClassVector>,
	substitutions: Mutex<Vec<Substitution>>,
	clocks: Vec<Py<PyClock>>,
	transitions: Mutex<Vec<Transitions>>,
	tree: Py<PyTree>,

	weights: Vec<u32>,
	likelihoods: Mutex<SliceBuffer<f64>>,
	selector: MonotonicBool,

	pool: Mutex<ThreadPool>,

	cache: MonotonicF64,
	last: MonotonicF64,
	launched_update: MonotonicBool,
}

#[pymethods]
impl HeteroLikelihood {
	#[new]
	fn new(
		msa: Py<PyMsa>,
		tree: Py<PyTree>,
		categories: Py<PyClassVector>,
		substitutions: Vec<Substitution>,
		clocks: Vec<Py<PyClock>>,
		calculator: CalculatorConfig,
	) -> Result<Self> {
		let (samples, weights) = deduplicate(msa.get());

		let num_categories = clocks.len();
		let num_patterns = weights.len();
		let likelihoods =
			SliceBuffer::new(num_patterns, num_categories * 2);

		let mut calculators = Vec::new();
		let mut transitions = Vec::new();

		let num_nodes = tree.get().num_nodes();
		for _ in 0..num_categories {
			calculators.push(calculator
				.make4(samples.clone(), weights.clone())?);

			let transition = Transitions::new(4, num_nodes);
			transitions.push(transition);
		}

		let pool = ThreadPool::try_spawn(num_categories)?;

		let out = Self {
			calculators: Mutex::new(calculators),

			categories,
			clocks,
			substitutions: Mutex::new(substitutions),
			transitions: Mutex::new(transitions),
			tree,

			weights,
			likelihoods: Mutex::new(likelihoods),
			selector: false.into(),
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
		let mut substitutions = self.substitutions.lock();

		let num_categories = transitions.len();

		let categories = self.categories.get().inner();

		for clock in &self.clocks {
			let mut clock = clock.get().inner();
			clock.update()?;
			clock.mark_tree(&mut tree);
		}

		for substitution in substitutions.iter_mut() {
			if substitution.update()? {
				tree.mark_all_edges_updated();
			}
		}

		if !tree.is_changed() && !categories.is_changed() {
			self.last.store(self.cache.load());
			return Ok(self.cache.load());
		}

		if tree.is_changed() {
			for ((transition, clock), substitution) in transitions
				.iter_mut()
				.zip(&self.clocks)
				.zip(substitutions.iter_mut())
			{
				let clock = clock.get().inner();
				transition.update(
					&mut tree,
					&mut **substitution,
					|edge| clock.get_rate(edge),
				)?;
			}

			self.launched_update.store(true);

			let calculators = &mut *self.calculators.lock();
			let calc_ptr =
				SyncMutPtr::new(calculators.as_mut_ptr());

			let new_selector = !self.selector.load();
			self.selector.store(new_selector);
			let offset = num_categories * usize::from(new_selector);

			self.pool.lock().for_n(self.clocks.len(), |prong| {
				let i = prong.task_index;
				let transition = &transitions[i];

				// SAFETY: `i` is less thant `calculator.len()`,
				// all accesses will be disjoint due to `for_n`.
				let calculator =
					unsafe { &mut *calc_ptr.get(i) };

				calculator
					.likelihood(&tree, transition)
					.unwrap();
				let mut likelihoods = self.likelihoods.lock();
				likelihoods[offset + i].copy_from_slice(
					calculator.likelihoods(),
				);
			});
		}

		let likelihoods = self.likelihoods.lock();
		let offset = usize::from(self.selector.load()) * num_categories;
		let num_patterns = likelihoods[0].len();
		let mut out = 0.0;
		for i in 0..num_patterns {
			let category = usize::from(categories[i]);
			let likelihood = likelihoods[offset + category][i];
			out += likelihood * f64::from(self.weights[i]);
		}

		self.last.store(out);
		Ok(out)
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
			for substitution in self.substitutions.lock().iter_mut()
			{
				substitution.accept();
			}
			for clock in &self.clocks {
				clock.get().inner().accept();
			}
		}
		self.launched_update.store(false);
		Ok(())
	}

	pub fn reject(&self) -> Result<()> {
		let old_selector = !self.selector.load();
		self.selector.store(old_selector);

		if self.launched_update.load() {
			for calculator in self.calculators.lock().iter_mut() {
				calculator.reject()?;
			}
			for transition in self.transitions.lock().iter_mut() {
				transition.reject();
			}
			for substitution in self.substitutions.lock().iter_mut()
			{
				substitution.reject();
			}
			for clock in &self.clocks {
				clock.get().inner().reject();
			}
		}
		self.launched_update.store(false);
		Ok(())
	}
}
