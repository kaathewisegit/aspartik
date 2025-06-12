use anyhow::Result;
use log::{debug, trace};
use pyo3::prelude::*;
use pyo3::{conversion::FromPyObject, exceptions::PyTypeError};

use profiler::profile;
use util::{py_bail, py_call_method};

pub struct PyPrior {
	/// INVARIANT: the type has a `probability` method
	inner: PyObject,
}

impl PyPrior {
	pub fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	pub fn clone_ref(&self, py: Python) -> PyObject {
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
		let repr = obj.repr()?;
		if !obj.getattr("probability")?.is_callable() {
			py_bail!(
				PyTypeError,
				"Prior objects must have a `probability` method,
				which takes no arguments and returns a real
				number.  Instead got {repr}",
			);
		}

		let out = Self {
			inner: obj.clone().unbind(),
		};
		debug!(
			target: "b3::prior::extract_bound",
			repr:%, id = out.id();
			""
		);
		Ok(out)
	}
}
