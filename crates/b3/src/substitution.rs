use anyhow::{anyhow, Context, Result};
use linalg::RowMatrix;
use pyo3::prelude::*;
use pyo3::{conversion::FromPyObject, exceptions::PyTypeError};

use util::{py_bail, py_call_method};

pub struct PySubstitution<const N: usize> {
	inner: PyObject,
}

pub type Substitution<const N: usize> = RowMatrix<f64, N, N>;

impl<'py, const N: usize> FromPyObject<'py> for PySubstitution<N> {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		if !obj.getattr("get_matrix")?.is_callable() {
			py_bail!(PyTypeError, "Substitution model objects must have an `get_matrix` method which returns a substitution matrix");
		}

		let dimensions =
			obj.getattr("dimensions")?.extract::<usize>()?;
		if dimensions != N {
			py_bail!(PyTypeError, "Expected the substitution model to have {N} dimensions, got {dimensions}");
		}

		Ok(Self {
			inner: obj.clone().unbind(),
		})
	}
}

impl<const N: usize> PySubstitution<N> {
	pub fn get_matrix(&self, py: Python) -> Result<Substitution<N>> {
		let matrix = py_call_method!(py, self.inner, "get_matrix")?;

		type Matrix<const N: usize> = [[f64; N]; N];

		let matrix =
			matrix.extract::<Matrix<N>>(py).with_context(|| {
				anyhow!("Expected the substitution model to return a matrix {0}x{0}.", N)
			})?;
		let matrix = RowMatrix::from(matrix);

		Ok(matrix)
	}
}
