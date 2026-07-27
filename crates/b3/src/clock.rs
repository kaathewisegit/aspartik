use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::parameters::{Parameter, PyClassVector, PyReal, Tree};
use util::{
	atomic::{MonotonicBool, MonotonicF64},
	py_call_method,
};

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
	updated: MonotonicBool,
	rates: Vec<MonotonicF64>,
	rate_categories: Py<PyClassVector>,
	distribution: Py<PyAny>,
}

impl RelaxedClock {
	fn new(
		py: Python,
		rate_categories: Py<PyClassVector>,
		distribution: Py<PyAny>,
	) -> Result<Self> {
		let num_categories =
			rate_categories.get().inner().num_classes() as usize;
		let rates = vec![MonotonicF64::from(0.0); num_categories];
		let out = Self {
			updated: false.into(),
			rates,
			rate_categories,
			distribution,
		};
		out.update_rates(py)?;
		Ok(out)
	}

	fn update_rates(&self, py: Python) -> Result<()> {
		let num_categories =
			self.rate_categories.get().inner().num_classes()
				as usize;
		let step = 1.0 / (num_categories as f64);
		let mut quantile = step / 2.0;
		for i in 0..num_categories {
			let rate = py_call_method!(
				py,
				self.distribution,
				"inverse_cdf",
				quantile,
			)?;
			self.rates[i].store(rate.extract::<f64>(py)?);
			quantile += step;
		}
		Ok(())
	}

	fn update(&mut self, tree: &mut Tree) -> Result<()> {
		let updated = Python::attach(|py| -> Result<bool> {
			let dist_upd = py_call_method!(
				py,
				self.distribution,
				"is_changed"
			)?;
			let dist_upd: bool = dist_upd.extract(py)?;
			if dist_upd {
				self.update_rates(py)?;
				return Ok(true);
			}

			Ok(false)
		})?;

		if updated {
			self.updated.store(true);
			tree.mark_all_edges_updated();
			return Ok(());
		}

		let cats = self.rate_categories.get().inner();
		for &i in cats.changed_indices() {
			tree.mark_edge_updated(i);
		}

		Ok(())
	}

	fn get_rate(&self, edge: usize) -> f64 {
		let category =
			self.rate_categories.get().inner()[edge] as usize;
		self.rates[category].load()
	}

	fn accept(&mut self) {
		self.updated.store(false);
	}

	fn reject(&mut self) {
		if self.updated.load() {
			Python::attach(|py| {
				self.update_rates(py).unwrap();
			});
			self.updated.store(false);
		}
	}
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
			Clock::Relaxed(c) => {
				c.rates.iter().map(|v| v.load()).collect()
			}
		}
	}
}

impl PyClock {
	pub fn inner(&self) -> MutexGuard<'_, Clock> {
		self.inner.lock()
	}
}
