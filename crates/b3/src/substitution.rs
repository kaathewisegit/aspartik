#![allow(unused_variables)]

use anyhow::Result;
use pyo3::prelude::*;

use std::ops::{Deref, DerefMut};

use crate::parameters::{Parameter, PyReal, PyRealVector};
use linalg::{
	MatrixMut, MatrixRef,
	const_matrix::{from_diagonal, mul as cmul},
	eigen,
	lu::inverse,
	mul,
};

pub trait SubstitutionModel {
	fn update(&mut self) -> Result<bool>;

	fn write_transition(&mut self, distance: f64, dst: MatrixMut<'_, f64>);

	fn write_frequencies(&self, dst: &mut [f64]);

	fn accept(&mut self);

	fn reject(&mut self);
}

pub struct Substitution(Box<dyn SubstitutionModel + Send + Sync>);

impl<'py> FromPyObject<'_, 'py> for Substitution {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		let py = obj.py();

		Ok(Self(if let Ok(jc) = obj.extract::<Py<JC>>() {
			Box::new(jc.get().clone_ref(py))
		} else if let Ok(k80) = obj.extract::<Py<K80>>() {
			Box::new(k80.get().clone_ref(py))
		} else if let Ok(hky) = obj.extract::<Py<HKY>>() {
			Box::new(hky.get().clone_ref(py))
		} else if let Ok(gtr) = obj.extract::<Py<GTR>>() {
			Box::new(gtr.get().clone_ref(py))
		} else if let Ok(sub) = obj.extract::<Py<GenericSubstitution>>()
		{
			Box::new(sub.get().clone_ref(py))
		} else {
			unimplemented!("error");
		}))
	}
}

impl Deref for Substitution {
	type Target = dyn SubstitutionModel;

	fn deref(&self) -> &Self::Target {
		self.0.as_ref()
	}
}

impl DerefMut for Substitution {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.0.as_mut()
	}
}

type M4 = [[f64; 4]; 4];

/// Jukes-Cantor
///
/// A simple model with equal state transition rates.
///
/// Jukes and Cantor 1969, Evolution of Protein Molecules,
/// <https://doi.org/10.1016/b978-1-4832-3211-9.50009-7>.
#[derive(Debug, Clone, Copy)]
#[pyclass(module = "aspartik.b3.substitutions", frozen, from_py_object)]
pub struct JC;

#[pymethods]
impl JC {
	#[new]
	fn new() -> Self {
		Self
	}
}

impl JC {
	fn clone_ref(&self, _: Python) -> JC {
		JC
	}
}

impl SubstitutionModel for JC {
	fn update(&mut self) -> Result<bool> {
		Ok(false)
	}

	fn write_transition(&mut self, distance: f64, dst: MatrixMut<'_, f64>) {
		let exp = (-4.0 / 3.0 * distance).exp();

		let diagonal = 0.25 + 0.75 * exp;
		let other = 0.25 - 0.25 * exp;

		let out = [
			[diagonal, other, other, other],
			[other, diagonal, other, other],
			[other, other, diagonal, other],
			[other, other, other, diagonal],
		];
		dst.copy_from(MatrixRef::from_array(&out));
	}

	fn write_frequencies(&self, dst: &mut [f64]) {
		dst.fill(0.25)
	}

	fn accept(&mut self) {}
	fn reject(&mut self) {}
}

/// Kimura 80
///
/// Equal base frequencies (A/C/G/T) with different transition (keeps
/// purines/pyrimidines) and transversion (purine to pyrimidine and visa versa).
///
/// Kimura 1980, A simple method for estimating evolutionary rates of base
/// substitutions through comparative studies of nucleotide sequences,
/// <https://doi.org/10.1007/BF01731581>.
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct K80 {
	/// A transition is taken to be kappa times more likely than a
	/// transversion.
	kappa: Py<PyReal>,
	cached_kappa: f64,
}

#[pymethods]
impl K80 {
	#[new]
	fn new(kappa: Py<PyReal>) -> Self {
		let cached_kappa = kappa.get().inner().value();
		Self {
			kappa,
			cached_kappa,
		}
	}
}

impl K80 {
	fn clone_ref(&self, py: Python) -> K80 {
		K80 {
			kappa: self.kappa.clone_ref(py),
			cached_kappa: self.cached_kappa,
		}
	}
}

impl SubstitutionModel for K80 {
	fn update(&mut self) -> Result<bool> {
		let kappa = &*self.kappa.get().inner();

		if kappa.is_changed() {
			self.cached_kappa = kappa.value();
			Ok(true)
		} else {
			Ok(false)
		}
	}

	fn write_transition(&mut self, distance: f64, dst: MatrixMut<'_, f64>) {
		let kappa = self.cached_kappa;

		let frac1 = -4.0 / (kappa + 2.0);
		let frac2 = -(2.0 * kappa + 2.0) / (kappa + 2.0);

		let arg1 = 0.25 * (distance * frac1).exp();
		let arg2 = 0.5 * (distance * frac2).exp();

		let diagonal = 0.25 + arg1 + arg2;
		let transition = 0.25 + arg1 - arg2;
		let transversion = 0.25 - arg1;

		let out = [
			[diagonal, transversion, transition, transversion],
			[transversion, diagonal, transversion, transition],
			[transition, transversion, diagonal, transversion],
			[transversion, transition, transversion, diagonal],
		];
		dst.copy_from(MatrixRef::from_array(&out));
	}

	fn write_frequencies(&self, dst: &mut [f64]) {
		dst.fill(0.25)
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {
		self.cached_kappa = self.kappa.get().inner().value();
	}
}

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
	kappa: Py<PyReal>,
	/// DNA nucleotide frequencies
	///
	/// Must have the length of 4, with each element corresponding to
	/// Adenine, Cytosine, Guanine, and Thymine respectively.
	frequencies: Py<PyRealVector>,

	cached_kappa: f64,
	cached_frequencies: [f64; 4],

	p: M4,
	inv_p: M4,
	diag: [f64; 4],
}

#[pymethods]
impl HKY {
	#[new]
	fn new(frequencies: Py<PyRealVector>, kappa: Py<PyReal>) -> Self {
		assert_eq!(frequencies.get().inner().len(), 4);

		let mut out = Self {
			kappa,
			frequencies,

			cached_kappa: f64::NAN,
			cached_frequencies: [f64::NAN; 4],

			p: [[0.0; 4]; 4],
			inv_p: [[0.0; 4]; 4],
			diag: [0.0; 4],
		};
		out.update_matrices();
		out
	}
}

impl HKY {
	fn clone_ref(&self, py: Python<'_>) -> Self {
		Self {
			kappa: self.kappa.clone_ref(py),
			frequencies: self.frequencies.clone_ref(py),
			cached_kappa: self.cached_kappa,
			cached_frequencies: self.cached_frequencies,
			p: self.p,
			inv_p: self.inv_p,
			diag: self.diag,
		}
	}

	fn update_matrices(&mut self) {
		self.cached_frequencies = {
			let freqs = &*self.frequencies.get().inner();
			[freqs[0], freqs[1], freqs[2], freqs[3]]
		};
		self.cached_kappa = self.kappa.get().inner().value();

		let kappa = self.cached_kappa;
		let [a, c, g, t] = self.cached_frequencies;
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

impl SubstitutionModel for HKY {
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

	fn write_transition(&mut self, distance: f64, dst: MatrixMut<'_, f64>) {
		let diag: M4 =
			from_diagonal(&self.diag.map(|v| (v * distance).exp()));

		let out = cmul(&cmul(&self.p, &diag), &self.inv_p);
		dst.copy_from(MatrixRef::from_array(&out));
	}

	fn write_frequencies(&self, dst: &mut [f64]) {
		dst.copy_from_slice(&self.cached_frequencies);
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {
		self.update_matrices();
	}
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct GTR {
	/// DNA nucleotide frequencies
	frequencies: Py<PyRealVector>,

	rates: Py<PyRealVector>,

	p: M4,
	inv_p: M4,
	diag: [f64; 4],

	has_changed: bool,
}

#[pymethods]
impl GTR {
	#[new]
	fn new(frequencies: Py<PyRealVector>, rates: Py<PyRealVector>) -> Self {
		assert_eq!(frequencies.get().inner().len(), 4);
		assert_eq!(rates.get().inner().len(), 6);

		let mut out = Self {
			frequencies,
			rates,

			p: [[0.0; 4]; 4],
			inv_p: [[0.0; 4]; 4],
			diag: [0.0; 4],

			has_changed: false,
		};
		out.update_matrices();
		out
	}
}

impl GTR {
	fn clone_ref(&self, py: Python<'_>) -> Self {
		Self {
			frequencies: self.frequencies.clone_ref(py),
			rates: self.rates.clone_ref(py),
			p: self.p,
			inv_p: self.inv_p,
			diag: self.diag,
			has_changed: self.has_changed,
		}
	}

	fn get_rates(&self) -> [f64; 6] {
		let rates = &*self.rates.get().inner();
		[rates[0], rates[1], rates[2], rates[3], rates[4], rates[5]]
	}

	fn get_frequencies(&self) -> [f64; 4] {
		let freqs = &*self.frequencies.get().inner();
		[freqs[0], freqs[1], freqs[2], freqs[3]]
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
		for row in &mut gtr {
			for element in row {
				*element /= div;
			}
		}

		let mut imaginary = [0.0; 4];
		eigen(&gtr, &mut self.diag, &mut imaginary, &mut self.p);
		inverse(&self.p, &mut self.inv_p);
	}
}

impl SubstitutionModel for GTR {
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

	fn write_transition(&mut self, distance: f64, dst: MatrixMut<'_, f64>) {
		let diag: M4 =
			from_diagonal(&self.diag.map(|v| (v * distance).exp()));

		let out = cmul(&cmul(&self.p, &diag), &self.inv_p);
		dst.copy_from(MatrixRef::from_array(&out));
	}

	fn write_frequencies(&self, dst: &mut [f64]) {
		dst.copy_from_slice(&self.get_frequencies());
	}

	fn accept(&mut self) {
		self.has_changed = false;
	}

	fn reject(&mut self) {
		if self.has_changed {
			self.update_matrices();
		}
	}
}

#[derive(Debug)]
#[pyclass(module = "aspartik.b3.substitutions", frozen)]
pub struct GenericSubstitution {
	frequencies: Py<PyRealVector>,
	rates: Py<PyRealVector>,

	p: Vec<f64>,
	inv_p: Vec<f64>,
	eigenvalues: Vec<f64>,
	has_changed: bool,
}

#[pymethods]
impl GenericSubstitution {
	#[new]
	fn new(frequencies: Py<PyRealVector>, rates: Py<PyRealVector>) -> Self {
		let n = frequencies.get().inner().len();
		let expected_rates = n * (n - 1) / 2;
		assert_eq!(rates.get().inner().len(), expected_rates);

		let mut out = Self {
			frequencies,
			rates,
			p: vec![0.0; n * n],
			inv_p: vec![0.0; n * n],
			eigenvalues: vec![0.0; n],
			has_changed: false,
		};
		out.update_matrices();
		out
	}
}

impl GenericSubstitution {
	fn clone_ref(&self, py: Python<'_>) -> Self {
		Self {
			frequencies: self.frequencies.clone_ref(py),
			rates: self.rates.clone_ref(py),
			p: self.p.clone(),
			inv_p: self.inv_p.clone(),
			eigenvalues: self.eigenvalues.clone(),
			has_changed: self.has_changed,
		}
	}

	fn update_matrices(&mut self) {
		let n = self.eigenvalues.len();
		let freqs = self.frequencies.get().inner();
		let rates = self.rates.get().inner();

		let mut q = vec![0.0; n * n];
		let mut q_ref = MatrixMut::from_slice(&mut q, n, n);

		let mut rate_idx = 0;
		// upper triangular
		for i in 0..n {
			for j in (i + 1)..n {
				q_ref[(i, j)] = freqs[j] * rates[rate_idx];
				rate_idx += 1;
			}
		}

		let mut rate_idx = 0;
		// lower triangular
		for i in 0..n {
			for j in 0..i {
				q_ref[(i, j)] = freqs[j] * rates[rate_idx];
				rate_idx += 1;
			}
		}

		let mut scale = 0.0;
		// diagonal
		for i in 0..n {
			let mut sum = 0.0;
			for j in 0..n {
				if j == i {
					continue;
				}
				sum += q_ref[(i, j)];
			}
			q_ref[(i, i)] = -sum;
			scale += sum * freqs[i];
		}

		for val in q.iter_mut() {
			*val /= scale;
		}

		let mut imaginary = vec![0.0; n];
		let q_ref = MatrixRef::from_slice(&q, n, n);
		let p_ref = MatrixMut::from_slice(&mut self.p, n, n);
		eigen(q_ref, &mut self.eigenvalues, &mut imaginary, p_ref);

		let p_ref = MatrixRef::from_slice(&self.p, n, n);
		let inv_p_ref = MatrixMut::from_slice(&mut self.inv_p, n, n);
		inverse(p_ref, inv_p_ref);
	}
}

impl SubstitutionModel for GenericSubstitution {
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

	// TODO: get rid of allocations
	fn write_transition(&mut self, distance: f64, dst: MatrixMut<'_, f64>) {
		let n = self.eigenvalues.len();
		assert!((*dst).is_square());
		assert_eq!((*dst).num_cols(), n);

		let mut inter = vec![0.0; n * n];
		let mut inter_ref = MatrixMut::from_slice(&mut inter, n, n);

		let p_ref = MatrixRef::from_slice(&self.p, n, n);
		let inv_p_ref = MatrixRef::from_slice(&self.inv_p, n, n);

		let mut diag = vec![0.0; n * n];
		let mut diag_ref = MatrixMut::from_slice(&mut diag, n, n);
		for i in 0..n {
			diag_ref[(i, i)] =
				(self.eigenvalues[i] * distance).exp();
		}

		mul(p_ref, *diag_ref, inter_ref.reborrow());
		mul(*inter_ref, inv_p_ref, dst);
	}

	fn write_frequencies(&self, dst: &mut [f64]) {
		let frequencies = self.frequencies.get().inner();
		for i in 0..frequencies.len() {
			dst[i] = frequencies[i];
		}
	}

	fn accept(&mut self) {
		self.has_changed = false;
	}

	fn reject(&mut self) {
		if self.has_changed {
			self.update_matrices();
			self.has_changed = false;
		}
	}
}
