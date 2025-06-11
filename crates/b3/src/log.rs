use anyhow::Result;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::mcmc::Mcmc;
use util::{py_bail, py_call_method};

pub struct PyLogger {
	inner: PyObject,
	every: usize,
}

impl<'py> FromPyObject<'py> for PyLogger {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		if !obj.getattr("log")?.is_callable() {
			py_bail!(
				PyTypeError,
				"Loggers must have a callable `log` method"
			);
		}

		let every = obj.getattr("every")?.extract::<usize>()?;

		Ok(PyLogger {
			inner: obj.clone().unbind(),
			every,
		})
	}
}

impl PyLogger {
	pub fn clone_ref(&self, py: Python) -> PyObject {
		self.inner.clone_ref(py)
	}

	pub fn should_log(&self, current_step: usize) -> bool {
		current_step % self.every == 0
	}

	pub fn log(&self, py: Python, mcmc: Py<Mcmc>) -> Result<()> {
		py_call_method!(py, self.inner, "log", mcmc)?;

		Ok(())
	}
}
