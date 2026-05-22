use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::parameters::{Parameter, PyReal, Tree};

pub struct StrictClock {
	rate: Py<PyReal>,
	cached_rate: f64,
	full_update: bool,
}

impl StrictClock {
	fn update(&mut self) -> Result<()> {
		if self.rate.get().inner().is_changed() {
			self.update_rate();
			self.full_update = true;
		}

		Ok(())
	}

	fn update_rate(&mut self) {
		self.cached_rate = self.rate.get().inner().value();
	}

	fn get_rate(&self) -> f64 {
		self.cached_rate
	}

	fn mark_tree(&self, tree: &mut Tree) {
		if self.full_update {
			tree.mark_all_edges_updated();
		}
	}

	fn accept(&mut self) {
		self.full_update = false;
	}

	fn reject(&mut self) {
		self.update_rate();
		self.full_update = false;
	}
}

pub enum Clock {
	Strict(StrictClock),
}

impl Clock {
	pub fn update(&mut self) -> Result<()> {
		match self {
			Self::Strict(clock) => clock.update(),
		}
	}

	pub fn get_rate(&self) -> f64 {
		match self {
			Self::Strict(clock) => clock.get_rate(),
		}
	}

	pub fn mark_tree(&self, tree: &mut Tree) {
		match self {
			Self::Strict(clock) => clock.mark_tree(tree),
		}
	}

	pub fn accept(&mut self) {
		match self {
			Self::Strict(clock) => clock.accept(),
		}
	}

	pub fn reject(&mut self) {
		match self {
			Self::Strict(clock) => clock.reject(),
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
		let strict = StrictClock {
			rate,
			cached_rate,
			full_update: true,
		};
		let clock = Clock::Strict(strict);
		Self {
			inner: Mutex::new(clock),
		}
	}
}

impl PyClock {
	pub fn inner(&self) -> MutexGuard<'_, Clock> {
		self.inner.lock()
	}
}
