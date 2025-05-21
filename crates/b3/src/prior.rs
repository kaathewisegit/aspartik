use anyhow::Result;
use pyo3::prelude::*;
use pyo3::{conversion::FromPyObject, exceptions::PyTypeError};
use tracing::{instrument, trace};

use util::{py_bail, py_call_method};

pub struct PyPrior {
	/// INVARIANT: the type has a `probability` method
	inner: PyObject,
}

impl PyPrior {
	fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	#[instrument(level = "trace", skip_all, fields(id = self.id()))]
	pub fn probability(&self, py: Python) -> Result<f64> {
		let out = py_call_method!(py, self.inner, "probability")?;
		let out = out.extract::<f64>(py)?;
		trace!(probability = out);
		Ok(out)
	}
}

impl<'py> FromPyObject<'py> for PyPrior {
	#[instrument(level = "trace", skip_all)]
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
		trace!(%repr, id = out.id(), "new PyPrior");
		Ok(out)
	}
}
