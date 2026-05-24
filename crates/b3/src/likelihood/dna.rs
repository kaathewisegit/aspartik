use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::deduplicate;
use crate::{
	Transitions,
	calculator::{Calculator, CalculatorConfig},
	clock::PyClock,
	parameters::{Parameter, PyTree},
	substitution::{PySubstitution4, SubstitutionModel},
};
use data::PyMsa;
use util::atomic::{MonotonicBool, MonotonicF64};

#[pyclass(name = "DNALikelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct DnaLikelihood {
	calculator: Mutex<Box<dyn Calculator<4, f64> + Send>>,

	substitution: Mutex<Box<dyn SubstitutionModel<4, f64> + Send>>,
	clock: Py<PyClock>,
	transitions: Mutex<Transitions<4, f64>>,
	tree: Py<PyTree>,

	cache: MonotonicF64,
	last: MonotonicF64,
	launched_update: MonotonicBool,
}

#[pymethods]
impl DnaLikelihood {
	#[new]
	fn new(
		msa: Py<PyMsa>,
		tree: Py<PyTree>,
		substitution: PySubstitution4,
		clock: Py<PyClock>,
		calculator: CalculatorConfig,
	) -> Result<Self> {
		let (samples, weights) = deduplicate(msa.get());
		let calculator = calculator.make4(samples, weights)?;

		let transitions = Transitions::new(tree.get().num_nodes());

		let out = Self {
			calculator: Mutex::new(calculator),

			substitution: Mutex::new(Box::new(substitution)),
			clock,
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

		self.transitions.lock().update(
			&mut tree,
			&**substitution,
			|edge| clock.get_rate(edge),
		)?;

		let frequencies = substitution.get_frequencies();
		drop(clock);
		drop(substitution);

		self.launched_update.store(true);
		let likelihood = self.calculator.lock().likelihood(
			tree,
			&self.transitions.lock(),
			frequencies,
		)?;
		self.last.store(likelihood);
		Ok(likelihood)
	}

	pub fn accept(&self) -> Result<()> {
		self.cache.store(self.last.load());
		if self.launched_update.load() {
			self.calculator.lock().accept()?;
			self.transitions.lock().accept();
			self.substitution.lock().accept();
			self.clock.get().inner().accept();
		}
		self.launched_update.store(false);
		Ok(())
	}

	pub fn reject(&self) -> Result<()> {
		if self.launched_update.load() {
			self.calculator.lock().reject()?;
			self.transitions.lock().reject();
			self.substitution.lock().reject();
			self.clock.get().inner().reject();
		}
		self.launched_update.store(false);
		Ok(())
	}
}
