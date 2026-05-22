use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::deduplicate;
use crate::{
	Transitions,
	calculator::{Calculator, CalculatorConfig},
	clock::PyClock,
	parameters::{Parameter, PyReal, PyTree},
	substitution::PySubstitution4,
	substitution::SubstitutionModel,
};
use data::PyMsa;
use util::atomic::{MonotonicBool, MonotonicF64};

#[pyclass(module = "aspartik.b3.likelihoods", frozen)]
pub struct GammaLikelihood {
	calculators: Mutex<Vec<Box<dyn Calculator<4, f64> + Send>>>,

	substitution: Mutex<Box<dyn SubstitutionModel<4, f64> + Send>>,
	clock: Py<PyClock>,
	transitions: Mutex<Vec<Transitions<4, f64>>>,
	tree: Py<PyTree>,

	cache: MonotonicF64,
	last: MonotonicF64,
	launched_update: MonotonicBool,
}

#[pymethods]
impl GammaLikelihood {
	#[new]
	#[expect(unused)]
	// TODO: roll num_categories and alpha into the clock
	#[expect(clippy::too_many_arguments)]
	fn new(
		py: Python,
		msa: Py<PyMsa>,
		tree: Py<PyTree>,
		substitution: PySubstitution4,
		num_categories: usize,
		clock_rate: Py<PyReal>,
		alpha: Py<PyReal>,
		calculator: CalculatorConfig,
	) -> Result<Self> {
		let (samples, weights) = deduplicate(msa.get());

		let mut calculators = Vec::new();
		let mut transitions = Vec::new();

		let num_nodes = tree.get().num_nodes();

		for _ in 0..num_categories {
			calculators.push(calculator.make4(samples, weights)?);

			let clock = todo!();
			let transition = Transitions::new(num_nodes);
			transitions.push(transition);
		}

		let out = Self {
			calculators: Mutex::new(calculators),

			clock: todo!(),
			substitution: Mutex::new(Box::new(substitution)),
			transitions: Mutex::new(transitions),
			tree,

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

		clock.update()?;
		clock.mark_tree(&mut tree);

		if substitution.update()? {
			tree.mark_all_edges_updated();
		}

		// no tree update, return the cache
		if !tree.is_changed() {
			self.last.store(self.cache.load());
			return Ok(self.cache.load());
		}

		for transition in transitions.iter_mut() {
			transition.update(
				&mut tree,
				&clock,
				&**substitution,
			)?;
		}

		let frequencies = substitution.get_frequencies();
		drop(clock);
		drop(substitution);
		drop(tree);

		self.launched_update.store(true);

		let mut calculators = self.calculators.lock();
		let mut likelihood = 0.0;
		for (calculator, transition) in
			calculators.iter_mut().zip(transitions.iter())
		{
			let tree = self.tree.get().inner();
			likelihood += calculator.likelihood(
				tree,
				transition,
				frequencies,
			)?;
		}

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
