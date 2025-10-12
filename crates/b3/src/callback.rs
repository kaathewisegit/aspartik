use anyhow::Result;
use pyo3::prelude::*;

use crate::mcmc::Mcmc;
use util::{py_bail, py_call_method, py_check_method, py_extract_attr};

pub struct PyCallback {
	inner: Py<PyAny>,
	every: usize,
}

impl<'py> FromPyObject<'py> for PyCallback {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "call");

		let every = py_extract_attr!(obj, "every", usize)?;

		Ok(PyCallback {
			inner: obj.clone().unbind(),
			every,
		})
	}
}

impl PyCallback {
	pub fn clone_ref(&self, py: Python) -> Py<PyAny> {
		self.inner.clone_ref(py)
	}

	pub fn should_call(&self, current_step: usize) -> bool {
		current_step.is_multiple_of(self.every)
	}

	pub fn call(&self, py: Python, mcmc: Py<Mcmc>) -> Result<()> {
		py_call_method!(py, self.inner, "log", mcmc)?;

		Ok(())
	}
}
