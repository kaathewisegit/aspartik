use anyhow::{Context, Result, anyhow};
use linalg::RowMatrix;
use log::debug;
use pyo3::prelude::*;
use pyo3::{conversion::FromPyObject, exceptions::PyValueError};

use util::{py_bail, py_call_method, py_check_method, py_extract_attr};

pub struct PySubstitution<const N: usize> {
	inner: Py<PyAny>,
}

pub type Substitution<const N: usize> = RowMatrix<f64, N, N>;

impl<'py, const N: usize> FromPyObject<'_, 'py> for PySubstitution<N> {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "get_matrix");

		let dimensions = py_extract_attr!(obj, "dimensions", usize)?;
		if dimensions != N {
			py_bail!(
				PyValueError,
				"Expected the substitution model to have {N} dimensions, got {dimensions}"
			);
		}

		let out = Self {
			inner: obj.to_owned().unbind(),
		};
		debug!(
			target: "b3::substitution::extract_bound",
			repr:% = obj.repr()?, id = out.id();
			""
		);
		Ok(out)
	}
}

impl<const N: usize> PySubstitution<N> {
	pub fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	pub fn get_matrix(&self, py: Python) -> Result<Substitution<N>> {
		let matrix = py_call_method!(py, self.inner, "get_matrix")?;

		type Matrix<const N: usize> = [[f64; N]; N];

		let matrix =
			matrix.extract::<Matrix<N>>(py).with_context(|| {
				anyhow!(
					"Expected the substitution model to return a matrix {0}x{0}.",
					N
				)
			})?;
		let matrix = RowMatrix::from(matrix);

		Ok(matrix)
	}
}

trait SubstitutionTrait<const N: usize> {
	fn update(&mut self, py: Python) -> Result<bool>;

	fn get_transition(&self, distance: f64) -> Substitution<N>;
}

pub struct SubstitutionModel<const N: usize> {
	inner: Box<dyn SubstitutionTrait<N> + Send>,
}

impl<const N: usize> SubstitutionModel<N> {
	pub fn update(&mut self, py: Python) -> Result<bool> {
		self.inner.update(py)
	}

	pub fn get_transition(&self, distance: f64) -> Substitution<N> {
		self.inner.get_transition(distance)
	}
}

impl<'py> FromPyObject<'_, 'py> for SubstitutionModel<4> {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
		let k80 = obj.cast::<K80>()?;
		let k80 = k80.get().clone(obj.py());

		Ok(Self {
			inner: Box::new(k80),
		})
	}
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct K80 {
	kappa: Py<PyAny>,
	cached_kappa: f64,
}

impl K80 {
	fn clone(&self, py: Python) -> Self {
		Self {
			kappa: self.kappa.clone_ref(py),
			cached_kappa: self.cached_kappa,
		}
	}
}

impl SubstitutionTrait<4> for K80 {
	fn update(&mut self, py: Python) -> Result<bool> {
		let kappa = self.kappa.extract(py)?;

		if kappa != self.cached_kappa {
			self.cached_kappa = kappa;
			Ok(true)
		} else {
			Ok(false)
		}
	}

	fn get_transition(&self, distance: f64) -> Substitution<4> {
		let kappa = self.cached_kappa;

		let frac1 = -4.0 / (kappa + 2.0);
		let frac2 = -(2.0 * kappa + 2.0) / (kappa + 2.0);

		let arg1 = 0.25 * (distance * frac1).exp();
		let arg2 = 0.5 * (distance * frac2).exp();

		let diagonal = 0.25 + arg1 + arg2;
		let transition = 0.25 + arg1 - arg2;
		let transversion = 0.25 - arg1;

		RowMatrix::from([
			[diagonal, transversion, transition, transversion],
			[transversion, diagonal, transversion, transition],
			[transition, transversion, diagonal, transversion],
			[transversion, transition, transversion, diagonal],
		])
	}
}

#[pymethods]
impl K80 {
	#[new]
	pub fn new(kappa: Py<PyAny>) -> Self {
		Self {
			kappa,
			cached_kappa: f64::NAN,
		}
	}
}
