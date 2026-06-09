use anyhow::Result;
use pyo3::prelude::*;

use crate::mcmc::Mcmc;
use util::{py_call_method, py_check_method, py_extract_attr, py_has_method};

mod op_stats;
mod trace_writer;

pub use op_stats::OperatorStats;
pub use trace_writer::TraceWriter;

pub struct PyCallback {
	inner: Py<PyAny>,
	every: usize,
}

impl<'py> FromPyObject<'_, 'py> for PyCallback {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "call");

		let every = py_extract_attr!(obj, "every", usize)?;

		Ok(PyCallback {
			inner: obj.to_owned().unbind(),
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
		py_call_method!(py, self.inner, "call", mcmc)?;

		Ok(())
	}

	pub fn finish(&self, py: Python, mcmc: Py<Mcmc>) -> Result<()> {
		if py_has_method!(self.inner.bind(py), "finish") {
			py_call_method!(py, self.inner, "finish", mcmc)?;
		}

		Ok(())
	}
}
