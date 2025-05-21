use anyhow::Result;
use linalg::RowMatrix;
use pyo3::prelude::*;
use pyo3::{conversion::FromPyObject, exceptions::PyTypeError};
use tracing::{instrument, trace};

use util::{py_bail, py_call_method};

pub struct PyClock {
	inner: PyObject,
}

pub type Substitution = RowMatrix<f64, 4, 4>;

impl<'py> FromPyObject<'py> for PyClock {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		let repr = obj.repr()?;
		if !obj.getattr("update")?.is_callable() {
			py_bail!(PyTypeError, "Substitution model objects must have an `update` method, which takes a list of edges and returns clock rates on these edges.  Instead got {repr}");
		}

		let out = Self {
			inner: obj.clone().unbind(),
		};
		trace!(%repr, id = out.id(), "new PyClock");
		Ok(out)
	}
}

impl PyClock {
	fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	#[instrument(skip_all, fields(id = self.id()))]
	pub fn update(
		&self,
		py: Python,
		edges: Vec<usize>,
	) -> Result<Vec<f64>> {
		let rates = py_call_method!(py, self.inner, "update", edges)?;
		let rates = rates.extract::<Vec<f64>>(py)?;

		Ok(rates)
	}
}
