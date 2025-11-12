use pyo3::{exceptions::PyTypeError, prelude::*};

use util::{py_bail, py_has_method};

#[derive(Debug)]
#[repr(transparent)]
pub struct SupportsFloat(Py<PyAny>);

impl<'py> FromPyObject<'_, 'py> for SupportsFloat {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		if !py_has_method!(obj, "__float__") {
			py_bail!(
				PyTypeError,
				"doesn't implement the `SupportsFloat` protocol"
			)
		}

		Ok(SupportsFloat(obj.to_owned().unbind()))
	}
}

impl<'py> IntoPyObject<'py> for SupportsFloat {
	type Target = PyAny;
	type Output = Bound<'py, PyAny>;
	type Error = PyErr;

	fn into_pyobject(
		self,
		py: Python<'py>,
	) -> Result<Self::Output, Self::Error> {
		Ok(self.0.bind(py).clone())
	}
}

impl SupportsFloat {
	pub fn extract(&self, py: Python) -> PyResult<f64> {
		self.0.extract(py)
	}

	pub fn clone_ref(&self, py: Python) -> Self {
		SupportsFloat(self.0.clone_ref(py))
	}
}
