use anyhow::Result;
use log::{debug, trace};
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;

use profiler::profile;
use util::{py_bail, py_call_method, py_check_method};

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
		let out = profile!(
			target: "b3::prior::probability"
			id = self.id();
			py_call_method!(py, self.inner, "probability")?
		);
		let out = out.extract::<f64>(py)?;
		trace!(target: "b3::prior", probability = out; "");
		Ok(out)
	}
}

impl<'py> FromPyObject<'py> for PyPrior {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "probability");

		let out = Self {
			inner: obj.clone().unbind(),
		};
		debug!(
			target: "b3::prior::extract_bound",
			repr:% = obj.repr()?, id = out.id();
			""
		);
		Ok(out)
	}
}
