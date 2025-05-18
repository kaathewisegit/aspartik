use anyhow::Result;
use pyo3::prelude::*;
use pyo3::{conversion::FromPyObject, exceptions::PyTypeError};

use util::{py_bail, py_call_method};

pub struct PyPrior {
	/// INVARIANT: the type has a `probability` method
	inner: PyObject,
}

impl PyPrior {
	pub fn probability(&self, py: Python) -> Result<f64> {
		let out = py_call_method!(py, self.inner, "probability")?;
		Ok(out.extract::<f64>(py)?)
	}
}

impl<'py> FromPyObject<'py> for PyPrior {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		if !obj.getattr("probability")?.is_callable() {
			py_bail!(
				PyTypeError,
				"Prior objects must have a `probability` method,
				which takes no arguments and returns a real
				number",
			);
		}

		Ok(Self {
			inner: obj.clone().unbind(),
		})
	}
}
