use anyhow::Result;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;

use util::{
	atomic::MonotonicF64, py_call_method, py_check_method, py_has_method,
};

mod coalescent;
mod dirichlet;
mod monophyly;
mod skyline;
mod yule;

pub use coalescent::{ConstantPopulation, ExponentialGrowth};
pub use dirichlet::SymmetricDirichlet;
pub use monophyly::Monophyly;
pub use skyline::BayesianSkyline;
pub use yule::Yule;

pub struct PyPrior {
	/// INVARIANT: the type methods `probability` and `is_changed`
	inner: Py<PyAny>,
	is_stateful: bool,

	last: MonotonicF64,
	cache: MonotonicF64,
}

impl PyPrior {
	pub fn clone_ref(&self, py: Python) -> Py<PyAny> {
		self.inner.clone_ref(py)
	}

	pub fn probability(&self, py: Python) -> Result<f64> {
		let is_changed = py_call_method!(py, self.inner, "is_changed")?
			.extract::<bool>(py)?;

		let out = if is_changed {
			let new_probability =
				py_call_method!(py, self.inner, "probability")?
					.extract::<f64>(py)?;
			self.last.store(new_probability);
			new_probability
		} else {
			self.last.store(self.cache.load());
			self.cache.load()
		};

		Ok(out)
	}

	pub fn accept(&self, py: Python) -> Result<()> {
		self.cache.store(self.last.load());
		if self.is_stateful {
			py_call_method!(py, self.inner, "accept")?;
		}
		Ok(())
	}

	pub fn reject(&self, py: Python) -> Result<()> {
		if self.is_stateful {
			py_call_method!(py, self.inner, "reject")?;
		}
		Ok(())
	}

	pub fn into_inner(self) -> Py<PyAny> {
		self.inner
	}
}

impl<'py> FromPyObject<'_, 'py> for PyPrior {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "probability");
		py_check_method!(obj, "is_changed");
		let is_stateful = py_has_method!(obj, "accept");

		let cache = py_call_method!(obj, "probability")?
			.extract::<f64>()?;

		Ok(Self {
			inner: obj.to_owned().unbind(),
			is_stateful,

			cache: cache.into(),
			last: f64::NAN.into(),
		})
	}
}
