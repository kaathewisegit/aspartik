use anyhow::{Context, Result, anyhow};
use linalg::{RowMatrix, Vector};
use log::debug;
use pyo3::{PyTypeCheck, prelude::*};
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

	fn get_frequencies(&self) -> Vector<f64, N>;
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

	pub fn get_frequencies(&self) -> Vector<f64, N> {
		self.inner.get_frequencies()
	}
}

impl<'py> FromPyObject<'_, 'py> for SubstitutionModel<4> {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
		if JC::type_check(&obj) {
			let jc = obj.cast::<JC>()?;
			let jc = *jc.get();

			Ok(Self {
				inner: Box::new(jc),
			})
		} else if K80::type_check(&obj) {
			let k80 = obj.cast::<K80>()?;
			let k80 = k80.get().clone(obj.py());

			Ok(Self {
				inner: Box::new(k80),
			})
		} else if HKY::type_check(&obj) {
			let hky = obj.cast::<HKY>()?;
			let hky = hky.get().clone(obj.py());

			Ok(Self {
				inner: Box::new(hky),
			})
		} else {
			todo!("Type error")
		}
	}
}

#[derive(Debug, Clone, Copy)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct JC;

impl SubstitutionTrait<4> for JC {
	fn update(&mut self, _py: Python) -> Result<bool> {
		Ok(false)
	}

	fn get_transition(&self, distance: f64) -> Substitution<4> {
		let exp = (-4.0 / 3.0 * distance).exp();

		let diagonal = 0.25 + 0.75 * exp;
		let other = 0.25 - 0.25 * exp;

		RowMatrix::from([
			[diagonal, other, other, other],
			[other, diagonal, other, other],
			[other, other, diagonal, other],
			[other, other, other, diagonal],
		])
	}

	fn get_frequencies(&self) -> Vector<f64, 4> {
		Vector::from_element(0.25)
	}
}

#[pymethods]
impl JC {
	#[new]
	fn new() -> Self {
		Self
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

	fn get_frequencies(&self) -> Vector<f64, 4> {
		Vector::from_element(0.25)
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

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct HKY {
	frequencies: Py<PyAny>,
	kappa: Py<PyAny>,

	cached_kappa: f64,
	cached_frequencies: (f64, f64, f64, f64),

	p: RowMatrix<f64, 4, 4>,
	inv_p: RowMatrix<f64, 4, 4>,
	diag: Vector<f64, 4>,
}

impl HKY {
	fn update_matrices(&mut self) {
		let kappa = self.cached_kappa;
		let (a, c, g, t) = self.cached_frequencies;
		let r = a + g;
		let y = c + t;

		self.p = RowMatrix::from([
			[1.0, -y / r, -g / a, 0.0],
			[1.0, 1.0, 0.0, -t / c],
			[1.0, -y / r, 1.0, 0.0],
			[1.0, 1.0, 0.0, 1.0],
		]);

		self.diag = [
			0.0,
			-1.0,
			-a * kappa - c - g * kappa - t,
			-a - c * kappa - g - t * kappa,
		]
		.into();

		self.inv_p = RowMatrix::from([
			[a, c, g, t],
			[-a, c * r / y, -g, t * r / y],
			[-a / r, 0.0, a / r, 0.0],
			[0.0, -c / y, 0.0, c / y],
		]);
	}

	fn clone(&self, py: Python) -> Self {
		Self {
			frequencies: self.frequencies.clone_ref(py),
			kappa: self.kappa.clone_ref(py),

			..*self
		}
	}
}

impl SubstitutionTrait<4> for HKY {
	fn update(&mut self, py: Python) -> Result<bool> {
		let frequencies =
			self.frequencies.extract::<(f64, f64, f64, f64)>(py)?;
		let kappa = self.kappa.extract::<f64>(py)?;

		if kappa != self.cached_kappa
			|| frequencies != self.cached_frequencies
		{
			self.cached_frequencies = frequencies;
			self.cached_kappa = kappa;
			self.update_matrices();
			Ok(true)
		} else {
			Ok(false)
		}
	}

	fn get_transition(&self, distance: f64) -> Substitution<4> {
		let diag = RowMatrix::from_diagonal(
			self.diag.map(|v| (v * distance).exp()),
		);

		self.p * diag * self.inv_p
	}

	fn get_frequencies(&self) -> Vector<f64, 4> {
		let array: [f64; 4] = self.cached_frequencies.into();
		array.into()
	}
}

#[pymethods]
impl HKY {
	#[new]
	pub fn new(frequencies: Py<PyAny>, kappa: Py<PyAny>) -> Self {
		Self {
			kappa,
			frequencies,

			cached_kappa: f64::NAN,
			cached_frequencies: (
				f64::NAN,
				f64::NAN,
				f64::NAN,
				f64::NAN,
			),

			p: RowMatrix::default(),
			inv_p: RowMatrix::default(),
			diag: Vector::default(),
		}
	}
}
