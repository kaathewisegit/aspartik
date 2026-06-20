use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::parameters::{Parameter, PyClassVector, PyReal, Tree};
use util::py_call_method;

pub struct StrictClock {
	rate: Py<PyReal>,
	cached_rate: f64,
}

impl StrictClock {
	fn update(&mut self, tree: &mut Tree) -> Result<()> {
		if self.rate.get().inner().is_changed() {
			self.update_rate();
			tree.mark_all_edges_updated();
		}

		Ok(())
	}

	fn update_rate(&mut self) {
		self.cached_rate = self.rate.get().inner().value();
	}

	fn get_rate(&self) -> f64 {
		self.cached_rate
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {
		self.update_rate();
	}
}

pub struct RelaxedClock {
	rates: Vec<f64>,
	rate_categories: Py<PyClassVector>,
}

impl RelaxedClock {
	fn new(
		py: Python,
		rate_categories: Py<PyClassVector>,
		distribution: Py<PyAny>,
	) -> Result<Self> {
		let num_categories =
			rate_categories.get().inner().num_classes() as usize;
		let mut rates = vec![0.0; num_categories];
		#[expect(clippy::needless_range_loop)]
		for i in 0..num_categories {
			let rate = py_call_method!(
				py,
				distribution,
				"inverse_cdf",
				(i + 1) as f64 / (num_categories + 1) as f64,
			)?;
			rates[i] = rate.extract::<f64>(py)?;
		}
		Ok(Self {
			rates,
			rate_categories,
		})
	}

	fn update(&mut self, tree: &mut Tree) -> Result<()> {
		let cats = self.rate_categories.get().inner();
		for i in 0..tree.num_edges() {
			if cats.is_changed_at(i) {
				tree.mark_edge_updated(i);
			}
		}

		Ok(())
	}

	fn get_rate(&self, edge: usize) -> f64 {
		let category =
			self.rate_categories.get().inner()[edge] as usize;
		self.rates[category]
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {}
}

pub enum Clock {
	Strict(StrictClock),
	Relaxed(RelaxedClock),
}

impl Clock {
	pub fn update(&mut self, tree: &mut Tree) -> Result<()> {
		match self {
			Self::Strict(clock) => clock.update(tree),
			Self::Relaxed(clock) => clock.update(tree),
		}
	}

	pub fn get_rate(&self, edge: usize) -> f64 {
		match self {
			Self::Strict(clock) => clock.get_rate(),
			Self::Relaxed(clock) => clock.get_rate(edge),
		}
	}

	pub fn accept(&mut self) {
		match self {
			Self::Strict(clock) => clock.accept(),
			Self::Relaxed(clock) => clock.accept(),
		}
	}

	pub fn reject(&mut self) {
		match self {
			Self::Strict(clock) => clock.reject(),
			Self::Relaxed(clock) => clock.reject(),
		}
	}
}

#[pyclass(name = "Clock", module = "aspartik.b3", frozen)]
pub struct PyClock {
	inner: Mutex<Clock>,
}

#[pymethods]
impl PyClock {
	#[pyo3(name = "Strict")]
	#[classmethod]
	fn strict(_cls: Py<PyType>, rate: Py<PyReal>) -> Self {
		let cached_rate = rate.get().inner().value();
		let strict = StrictClock { rate, cached_rate };
		let clock = Clock::Strict(strict);
		Self {
			inner: Mutex::new(clock),
		}
	}

	#[pyo3(name = "Relaxed")]
	#[classmethod]
	fn relaxed(
		_cls: Py<PyType>,
		py: Python,
		rate_categories: Py<PyClassVector>,
		distribution: Py<PyAny>,
	) -> Result<Self> {
		let relaxed =
			RelaxedClock::new(py, rate_categories, distribution)?;
		let clock = Clock::Relaxed(relaxed);
		Ok(Self {
			inner: Mutex::new(clock),
		})
	}

	/// All of the clock rates represented by this clock
	///
	/// This is a list for the relaxed clock and a unit list `[rate]` for a
	/// strict clock.
	#[getter]
	fn rates(&self) -> Vec<f64> {
		match &*self.inner() {
			Clock::Strict(c) => vec![c.rate.get().inner().value()],
			Clock::Relaxed(c) => c.rates.clone(),
		}
	}
}

impl PyClock {
	pub fn inner(&self) -> MutexGuard<'_, Clock> {
		self.inner.lock()
	}
}
