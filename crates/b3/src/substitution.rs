use anyhow::Result;
use pyo3::prelude::*;

use crate::parameters::{Parameter, PyReal, PyRealVector};
use linalg::{ConstMatrix, ConstSquareMatrix, eigen, lu::inverse};

pub trait SubstitutionModel<const N: usize, F> {
	fn update(&mut self) -> Result<bool>;

	fn get_transition(&self, distance: f64) -> [[F; N]; N];

	fn get_frequencies(&self) -> [F; N];

	fn accept(&mut self);

	fn reject(&mut self);
}

#[derive(FromPyObject, IntoPyObject)]
pub enum PySubstitution4 {
	JC(Py<PyJC>),
	K80(Py<PyK80>),
	HKY(Py<PyHKY>),
	GTR(Py<PyGTR>),
}

impl SubstitutionModel<4, f64> for PySubstitution4 {
	fn update(&mut self) -> Result<bool> {
		match self {
			Self::JC(m) => m.get().inner().update(),
			Self::K80(m) => m.get().inner().update(),
			Self::HKY(m) => m.get().inner().update(),
			Self::GTR(m) => m.get().inner().update(),
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
			Self::GTR(m) => {
				m.get().inner().get_transition(distance)
			}
		}
	}

	fn get_frequencies(&self) -> [f64; 4] {
		match self {
			Self::JC(m) => m.get().inner().get_frequencies(),
			Self::K80(m) => m.get().inner().get_frequencies(),
			Self::HKY(m) => m.get().inner().get_frequencies(),
			Self::GTR(m) => m.get().inner().get_frequencies(),
		}
	}

	fn accept(&mut self) {
		match self {
			Self::JC(m) => m.get().inner().accept(),
			Self::K80(m) => m.get().inner().accept(),
			Self::HKY(m) => m.get().inner().accept(),
			Self::GTR(m) => m.get().inner().accept(),
		}
	}

	fn reject(&mut self) {
		match self {
			Self::JC(m) => m.get().inner().reject(),
			Self::K80(m) => m.get().inner().reject(),
			Self::HKY(m) => m.get().inner().reject(),
			Self::GTR(m) => m.get().inner().reject(),
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

type M4 = [[f64; 4]; 4];

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

		[
			[diagonal, other, other, other],
			[other, diagonal, other, other],
			[other, other, diagonal, other],
			[other, other, other, diagonal],
		]
	}

	fn get_frequencies(&self) -> [f64; 4] {
		[0.25; 4]
	}

	fn accept(&mut self) {}
	fn reject(&mut self) {}
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
		let cached_kappa = kappa.get().inner().value();
		Self {
			kappa,
			cached_kappa,
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

	fn get_transition(&self, distance: f64) -> M4 {
		let kappa = self.cached_kappa;

		let frac1 = -4.0 / (kappa + 2.0);
		let frac2 = -(2.0 * kappa + 2.0) / (kappa + 2.0);

		let arg1 = 0.25 * (distance * frac1).exp();
		let arg2 = 0.5 * (distance * frac2).exp();

		let diagonal = 0.25 + arg1 + arg2;
		let transition = 0.25 + arg1 - arg2;
		let transversion = 0.25 - arg1;

		[
			[diagonal, transversion, transition, transversion],
			[transversion, diagonal, transversion, transition],
			[transition, transversion, diagonal, transversion],
			[transversion, transition, transversion, diagonal],
		]
	}

	fn get_frequencies(&self) -> [f64; 4] {
		[0.25; 4]
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {
		self.cached_kappa = self.kappa.get().inner().value();
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

	p: M4,
	inv_p: M4,
	diag: [f64; 4],
}

impl HKY {
	fn new(frequencies: Py<PyRealVector>, kappa: Py<PyReal>) -> Self {
		assert_eq!(frequencies.get().inner().len(), 4);

		let mut out = Self {
			kappa,
			frequencies,

			cached_kappa: f64::NAN,
			cached_frequencies: (
				f64::NAN,
				f64::NAN,
				f64::NAN,
				f64::NAN,
			),

			p: M4::zeros(),
			inv_p: M4::zeros(),
			diag: [0.0; 4],
		};
		out.update_matrices();
		out
	}

	fn update_matrices(&mut self) {
		self.cached_frequencies = {
			let freqs = &*self.frequencies.get().inner();
			(freqs[0], freqs[1], freqs[2], freqs[3])
		};
		self.cached_kappa = self.kappa.get().inner().value();

		let kappa = self.cached_kappa;
		let (a, c, g, t) = self.cached_frequencies;
		let r = a + g;
		let y = c + t;

		self.p = [
			[1.0, -y / r, -g / a, 0.0],
			[1.0, 1.0, 0.0, -t / c],
			[1.0, -y / r, 1.0, 0.0],
			[1.0, 1.0, 0.0, 1.0],
		];

		let div = 2.0
			* (g * t + a * c
				+ a * t + c * g + kappa * (a * g + c * t));

		self.diag = [
			0.0,
			-1.0 / div,
			-(y + r * kappa) / div,
			-(r + y * kappa) / div,
		];

		self.inv_p = [
			[a, c, g, t],
			[-a, c * r / y, -g, t * r / y],
			[-a / r, 0.0, a / r, 0.0],
			[0.0, -c / y, 0.0, c / y],
		];
	}
}

impl SubstitutionModel<4, f64> for HKY {
	fn update(&mut self) -> Result<bool> {
		if self.kappa.get().inner().is_changed()
			|| self.frequencies.get().inner().is_changed()
		{
			self.update_matrices();
			Ok(true)
		} else {
			Ok(false)
		}
	}

	fn get_transition(&self, distance: f64) -> M4 {
		let diag: M4 = ConstSquareMatrix::from_diagonal(
			self.diag.map(|v| (v * distance).exp()),
		);

		self.p.mul::<_, M4>(&diag).mul(&self.inv_p)
	}

	fn get_frequencies(&self) -> [f64; 4] {
		self.cached_frequencies.into()
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {
		self.update_matrices();
	}
}

create_pysubstitution!(PyHKY, HKY, "HKY", frequencies: Py<PyRealVector>, kappa: Py<PyReal>);

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct GTR {
	/// DNA nucleotide frequencies
	#[pyo3(get)]
	frequencies: Py<PyRealVector>,

	#[pyo3(get)]
	rates: Py<PyRealVector>,

	p: M4,
	inv_p: M4,
	diag: [f64; 4],

	has_changed: bool,
}

impl GTR {
	fn new(frequencies: Py<PyRealVector>, rates: Py<PyRealVector>) -> Self {
		assert_eq!(frequencies.get().inner().len(), 4);
		assert_eq!(rates.get().inner().len(), 6);

		let mut out = Self {
			frequencies,
			rates,

			p: M4::zeros(),
			inv_p: M4::zeros(),
			diag: [0.0; 4],

			has_changed: false,
		};
		out.update_matrices();
		out
	}

	fn get_rates(&self) -> [f64; 6] {
		let rates = &*self.rates.get().inner();
		[rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]]
	}

	fn update_matrices(&mut self) {
		let [p_a, p_c, p_g, p_t] = self.get_frequencies();
		let [a, b, c, d, e, f] = self.get_rates();

		let mut gtr = [
			[
				-a * p_c - b * p_g - c * p_t,
				a * p_c,
				b * p_g,
				c * p_t,
			],
			[
				a * p_a,
				-a * p_a - d * p_g - e * p_t,
				d * p_g,
				e * p_t,
			],
			[
				b * p_a,
				d * p_c,
				-b * p_a - d * p_c - f * p_t,
				f * p_t,
			],
			[
				c * p_a,
				e * p_c,
				f * p_g,
				-c * p_a - e * p_c - f * p_g,
			],
		];
		let div = 2.0
			* (a * p_a * p_c
				+ b * p_a * p_g + c * p_a * p_t
				+ d * p_c * p_g + e * p_c * p_t
				+ f * p_g * p_t);
		gtr.for_each(|e| *e /= div);

		let mut imaginary = [0.0; 4];
		eigen(&gtr, &mut self.diag, &mut imaginary, &mut self.p);
		inverse(&self.p, &mut self.inv_p);
	}
}

impl SubstitutionModel<4, f64> for GTR {
	fn update(&mut self) -> Result<bool> {
		if self.frequencies.get().inner().is_changed()
			|| self.rates.get().inner().is_changed()
		{
			self.update_matrices();
			self.has_changed = true;
			Ok(true)
		} else {
			Ok(false)
		}
	}

	fn get_transition(&self, distance: f64) -> M4 {
		let diag: M4 = ConstSquareMatrix::from_diagonal(
			self.diag.map(|v| (v * distance).exp()),
		);

		self.p.mul::<_, M4>(&diag).mul(&self.inv_p)
	}

	fn get_frequencies(&self) -> [f64; 4] {
		let freqs = &*self.frequencies.get().inner();
		[freqs[0], freqs[1], freqs[2], freqs[3]]
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {
		if self.has_changed {
			self.update_matrices();
		}
	}
}

create_pysubstitution!(
	PyGTR, GTR, "GTR",
	frequencies: Py<PyRealVector>, rates: Py<PyRealVector>
);
