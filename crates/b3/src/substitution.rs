use anyhow::Result;
use linalg::{RowMatrix, Vector};
use pyo3::{conversion::FromPyObject, prelude::*};

use crate::parameters::{Parameter, PyReal, PyRealVector};

pub trait SubstitutionModel<const N: usize, F> {
	fn update(&mut self) -> Result<bool>;

	fn get_transition(&self, distance: f64) -> [[F; N]; N];

	fn get_frequencies(&self) -> [F; N];
}

pub enum PySubstitution4 {
	JC(Py<PyJC>),
	K80(Py<PyK80>),
	HKY(Py<PyHKY>),
}

impl<'py> FromPyObject<'_, 'py> for PySubstitution4 {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		if let Ok(class_vector) = obj.cast::<PyJC>() {
			Ok(Self::JC(class_vector.into()))
		} else if let Ok(real) = obj.cast::<PyK80>() {
			Ok(Self::K80(real.into()))
		} else if let Ok(real_vector) = obj.cast::<PyHKY>() {
			Ok(Self::HKY(real_vector.into()))
		} else {
			todo!("descriptive error")
		}
	}
}

impl<'py> IntoPyObject<'py> for PySubstitution4 {
	type Target = PyAny;
	type Output = Bound<'py, PyAny>;
	type Error = PyErr;

	fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, PyErr> {
		Ok(match self {
			Self::JC(p) => {
				Bound::new(py, p.clone_ref(py))?.into_any()
			}
			Self::K80(p) => {
				Bound::new(py, p.clone_ref(py))?.into_any()
			}
			Self::HKY(p) => {
				Bound::new(py, p.clone_ref(py))?.into_any()
			}
		})
	}
}

impl SubstitutionModel<4, f64> for PySubstitution4 {
	fn update(&mut self) -> Result<bool> {
		match self {
			Self::JC(m) => m.get().inner().update(),
			Self::K80(m) => m.get().inner().update(),
			Self::HKY(m) => m.get().inner().update(),
		}
	}

	fn get_transition(&self, distance: f64) -> [[f64; 4]; 4] {
		match self {
			Self::JC(m) => m.get().inner().get_transition(distance),
			Self::K80(m) => {
				m.get().inner().get_transition(distance)
			}
			Self::HKY(m) => {
				m.get().inner().get_transition(distance)
			}
		}
	}

	fn get_frequencies(&self) -> [f64; 4] {
		match self {
			Self::JC(m) => m.get().inner().get_frequencies(),
			Self::K80(m) => m.get().inner().get_frequencies(),
			Self::HKY(m) => m.get().inner().get_frequencies(),
		}
	}
}

macro_rules! create_pysubstitution {
	($pytype:tt, $type:ty, $str:literal, $($v:ident: $t:ty),*) => {
		#[pyclass(name = $str, module = "aspartik.b3.substitutions", frozen)]
		pub struct $pytype {
			inner: parking_lot::Mutex<$type>,
		}

		impl $pytype {
			pub fn inner(&self) -> parking_lot::MutexGuard<'_, $type> {
				self.inner.lock()
			}
		}

		#[pymethods]
		impl $pytype {
			#[new]
			fn new($($v: $t),*) -> Self {
				Self {
					inner: <$type>::new($($v),*).into(),
				}
			}
		}
	};
}

/// Jukes-Cantor
///
/// A simple model with equal state transition rates.
///
/// Jukes and Cantor 1969, Evolution of Protein Molecules,
/// <https://doi.org/10.1016/b978-1-4832-3211-9.50009-7>.
#[derive(Debug, Clone, Copy)]
pub struct JC;

impl JC {
	fn new() -> Self {
		Self
	}
}

impl SubstitutionModel<4, f64> for JC {
	fn update(&mut self) -> Result<bool> {
		Ok(false)
	}

	fn get_transition(&self, distance: f64) -> [[f64; 4]; 4] {
		let exp = (-4.0 / 3.0 * distance).exp();

		let diagonal = 0.25 + 0.75 * exp;
		let other = 0.25 - 0.25 * exp;

		RowMatrix::from([
			[diagonal, other, other, other],
			[other, diagonal, other, other],
			[other, other, diagonal, other],
			[other, other, other, diagonal],
		])
		.into()
	}

	fn get_frequencies(&self) -> [f64; 4] {
		[0.25; 4]
	}
}

create_pysubstitution!(PyJC, JC, "JC",);

/// Kimura 80
///
/// Equal base frequencies (A/C/G/T) with different transition (keeps
/// purines/pyrimidines) and transversion (purine to pyrimidine and visa versa).
///
/// Kimura 1980, A simple method for estimating evolutionary rates of base
/// substitutions through comparative studies of nucleotide sequences,
/// <https://doi.org/10.1007/BF01731581>.
#[derive(Debug)]
pub struct K80 {
	/// A transition is taken to be kappa times more likely than a
	/// transversion.
	kappa: Py<PyReal>,
	cached_kappa: f64,
}

impl K80 {
	fn new(kappa: Py<PyReal>) -> Self {
		Self {
			kappa,
			cached_kappa: f64::NAN,
		}
	}
}

impl SubstitutionModel<4, f64> for K80 {
	fn update(&mut self) -> Result<bool> {
		let kappa = &*self.kappa.get().inner();

		if kappa.is_changed() {
			self.cached_kappa = kappa.value();
			Ok(true)
		} else {
			Ok(false)
		}
	}

	fn get_transition(&self, distance: f64) -> [[f64; 4]; 4] {
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
		.into()
	}

	fn get_frequencies(&self) -> [f64; 4] {
		[0.25; 4]
	}
}

create_pysubstitution!(PyK80, K80, "K80", kappa: Py<PyReal>);

/// Hasegawa et al. 1985
///
/// A model which can be thought of as a combination of K80 and F81: both base
/// rates and transition/transversion ratio are configurable.
///
/// Hasegawa et al. 1985, Dating of the human-ape splitting by a molecular clock
/// of mitochondrial DNA, <https://doi.org/10.1007/BF02101694>.
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct HKY {
	/// Transition/transversion ratio
	#[pyo3(get)]
	kappa: Py<PyReal>,
	/// DNA nucleotide frequencies
	///
	/// Must have the length of 4, with each element corresponding to
	/// Adenine, Cytosine, Guanine, and Thymine respectively.
	#[pyo3(get)]
	frequencies: Py<PyRealVector>,

	cached_kappa: f64,
	cached_frequencies: (f64, f64, f64, f64),

	p: RowMatrix<f64, 4, 4>,
	inv_p: RowMatrix<f64, 4, 4>,
	diag: Vector<f64, 4>,
}

impl HKY {
	fn new(frequencies: Py<PyRealVector>, kappa: Py<PyReal>) -> Self {
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

		let div = 2.0
			* (g * t + a * c
				+ a * t + c * g + kappa * (a * g + c * t));

		self.diag = [
			0.0,
			-1.0 / div,
			-(y + r * kappa) / div,
			-(r + y * kappa) / div,
		]
		.into();

		self.inv_p = RowMatrix::from([
			[a, c, g, t],
			[-a, c * r / y, -g, t * r / y],
			[-a / r, 0.0, a / r, 0.0],
			[0.0, -c / y, 0.0, c / y],
		]);
	}
}

impl SubstitutionModel<4, f64> for HKY {
	fn update(&mut self) -> Result<bool> {
		let frequencies = {
			let freqs = &*self.frequencies.get().inner();
			(freqs[0], freqs[1], freqs[2], freqs[3])
		};

		if self.kappa.get().inner().is_changed()
			|| frequencies != self.cached_frequencies
		{
			self.cached_frequencies = frequencies;
			self.cached_kappa = self.kappa.get().inner().value();
			self.update_matrices();
			Ok(true)
		} else {
			Ok(false)
		}
	}

	fn get_transition(&self, distance: f64) -> [[f64; 4]; 4] {
		let diag = RowMatrix::from_diagonal(
			self.diag.map(|v| (v * distance).exp()),
		);

		(self.p * diag * self.inv_p).into()
	}

	fn get_frequencies(&self) -> [f64; 4] {
		self.cached_frequencies.into()
	}
}

create_pysubstitution!(PyHKY, HKY, "HKY", frequencies: Py<PyRealVector>, kappa: Py<PyReal>);
