use anyhow::Result;
use linalg::RowMatrix;
use log::debug;
use pyo3::prelude::*;
use pyo3::{conversion::FromPyObject, exceptions::PyTypeError};

use profiler::profile;
use util::{py_bail, py_call_method};

pub struct PyClock {
	inner: PyObject,
}

pub type Substitution = RowMatrix<f64, 4, 4>;

impl<'py> FromPyObject<'py> for PyClock {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		let repr = obj.repr()?;
		if !obj.getattr("get_rate")?.is_callable() {
			py_bail!(PyTypeError, "Substitution model objects must have a `get_rate` method, which returns the clock rate.  Instead got {repr}");
		}

		let out = Self {
			inner: obj.clone().unbind(),
		};
		debug!(
			target: "b3::clock::extract_bound",
			repr:%, id = out.id();
			""
		);
		Ok(out)
	}
}

impl PyClock {
	pub fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	pub fn get_rate(&self, py: Python) -> Result<f64> {
		let rate = profile!(
			target: "b3::operator::propose",
			id = self.id();
			py_call_method!(py, self.inner, "get_rate")?
		);
		let rate = rate.extract::<f64>(py)?;

		Ok(rate)
	}
}
