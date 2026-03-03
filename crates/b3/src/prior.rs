use anyhow::Result;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;

use util::{py_call_method, py_check_method, py_has_method};

pub struct PyPrior {
	/// INVARIANT: the type has a `probability` method
	inner: Py<PyAny>,
	is_stateful: bool,
}

impl PyPrior {
	pub fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	pub fn clone_ref(&self, py: Python) -> Py<PyAny> {
		self.inner.clone_ref(py)
	}

	pub fn probability(&self, py: Python) -> Result<f64> {
		let out = py_call_method!(py, self.inner, "probability")?;
		let out = out.extract::<f64>(py)?;
		Ok(out)
	}

	pub fn accept(&self, py: Python) -> Result<()> {
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
}

impl<'py> FromPyObject<'_, 'py> for PyPrior {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "probability");
		let is_stateful = py_has_method!(obj, "accept");

		Ok(Self {
			inner: obj.to_owned().unbind(),
			is_stateful,
		})
	}
}
