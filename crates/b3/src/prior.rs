use anyhow::Result;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;

use logger::{debug, trace};
use util::{py_call_method, py_check_method};

pub struct PyPrior {
	/// INVARIANT: the type has a `probability` method
	inner: Py<PyAny>,
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
		trace!(target: "b3::prior", probability = out);
		Ok(out)
	}
}

impl<'py> FromPyObject<'_, 'py> for PyPrior {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "probability");

		let out = Self {
			inner: obj.to_owned().unbind(),
		};
		let repr = obj.repr()?;
		debug!(
			target: "b3::prior::extract_bound",
			repr = repr.to_str()?, id = out.id()
		);
		Ok(out)
	}
}
