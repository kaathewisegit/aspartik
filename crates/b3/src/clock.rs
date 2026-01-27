use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::{prelude::*, types::PyType};

use crate::parameters::Tree;
use pyutil::SupportsFloat;

pub struct StrictClock {
	rate: SupportsFloat,
	cached_rate: f64,
	full_update: bool,
}

impl StrictClock {
	fn update(&mut self, py: Python) -> Result<()> {
		let rate = self.rate.extract(py)?;
		if rate != self.cached_rate {
			self.cached_rate = rate;
			self.full_update = true;
		} else {
			self.full_update = false;
		}

		Ok(())
	}

	fn get_rate(&self) -> f64 {
		self.cached_rate
	}

	fn mark_tree(&self, tree: &mut Tree) {
		if self.full_update {
			tree.mark_all_edges_updated();
		}
	}
}

pub enum Clock {
	Strict(StrictClock),
}

impl Clock {
	pub fn update(&mut self, py: Python) -> Result<()> {
		match self {
			Self::Strict(clock) => clock.update(py),
		}
	}

	pub fn get_rate(&self, _edge: usize) -> f64 {
		match self {
			Self::Strict(clock) => clock.get_rate(),
		}
	}

	pub fn mark_tree(&self, tree: &mut Tree) {
		match self {
			Self::Strict(clock) => clock.mark_tree(tree),
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
	fn strict(_cls: Py<PyType>, rate: SupportsFloat) -> Self {
		let strict = StrictClock {
			rate,
			cached_rate: f64::NAN,
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
