use anyhow::Result;
use log::debug;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;

use util::{py_call_method, py_check_method};

pub struct PyClock {
	inner: Py<PyAny>,
}

impl<'py> FromPyObject<'_, 'py> for PyClock {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "get_rate");

		let out = Self {
			inner: obj.to_owned().unbind(),
		};
		debug!(
			target: "b3::clock::extract_bound",
			repr:% = obj.repr()?, id = out.id();
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
		let rate = py_call_method!(py, self.inner, "get_rate")?;
		let rate = rate.extract::<f64>(py)?;

		Ok(rate)
	}
}
