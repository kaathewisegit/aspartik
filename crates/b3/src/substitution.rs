use anyhow::Result;
use pyo3::{exceptions::PyTypeError, prelude::*};

use std::ops::{Deref, DerefMut};

use crate::parameters::{Parameter, PyReal, PyRealVector};
use linalg::{
	MatrixArrayExt, MatrixMut, MatrixSliceExt, beast_eigen,
	const_matrix::{from_diagonal, mul as cmul},
	mul,
};
use util::py_bail;

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
			py_bail!(
				PyTypeError,
				"{} is not a Substitution",
				obj.get_type().name()?
			)
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
		dst.copy_from(out.as_mat());
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
		dst.copy_from(out.as_mat());
	}

	fn write_frequencies(&self, dst: &mut [f64]) {
		dst.fill(0.25)
	}

	fn accept(&mut self) {}

	fn reject(&mut self) {
		self.cached_kappa = self.kappa.get().inner().value();
	}
}

fn transition_freqs(freqs: &[f64; 4], mut dst: MatrixMut<'_, f64>) {
	for i in 0..4 {
		for j in 0..4 {
			dst[(i, j)] = freqs[j];
		}
	}
}

/// Transition matrix for small distances
///
/// `exp(X)` is defined as `∑1/n! Xⁿ`.  When `X = Q t`, where `t` is small, we
/// can discard all items past `n = 1`.  Thus, `exp(Q t) ≈ I + Q t`.
fn transition_q(q: &M4, distance: f64, mut dst: MatrixMut<'_, f64>) {
	for i in 0..4 {
		for j in 0..4 {
			dst[(i, j)] = q[i][j] * distance;
		}
	}
	for i in 0..4 {
		dst[(i, i)] = 1.0;
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

	q: M4,
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

			q: Default::default(),
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
			q: self.q,
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

		self.q = [
			[0.0, c, kappa * g, t],
			[a, 0.0, g, kappa * t],
			[kappa * a, c, 0.0, t],
			[a, kappa * c, g, 0.0],
		];

		let div = 2.0
			* (g * t + a * c
				+ a * t + c * g + kappa * (a * g + c * t));

		for i in 0..4 {
			self.q[i][i] = -self.q[i].iter().sum::<f64>();
			for j in 0..4 {
				self.q[i][j] /= div;
			}
		}

		self.p = [
			[1.0, -y / r, -g / a, 0.0],
			[1.0, 1.0, 0.0, -t / c],
			[1.0, -y / r, 1.0, 0.0],
			[1.0, 1.0, 0.0, 1.0],
		];

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
		if distance > 20.0 {
			transition_freqs(&self.cached_frequencies, dst);
			return;
		} else if distance < 1e-10 {
			transition_q(&self.q, distance, dst);
			return;
		}

		let diag: M4 =
			from_diagonal(&self.diag.map(|v| (v * distance).exp()));

		let out = cmul(&cmul(&self.p, &diag), &self.inv_p);
		dst.copy_from(out.as_mat());
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

	cached_frequencies: [f64; 4],

	p: M4,
	inv_p: M4,
	diag: [f64; 4],

	q: M4,

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

			cached_frequencies: [f64::NAN; 4],

			p: [[0.0; 4]; 4],
			inv_p: [[0.0; 4]; 4],
			diag: [0.0; 4],

			q: Default::default(),

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
			cached_frequencies: self.cached_frequencies,
			p: self.p,
			inv_p: self.inv_p,
			diag: self.diag,
			q: self.q,
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
		self.cached_frequencies = self.get_frequencies();
		let [p_a, p_c, p_g, p_t] = self.cached_frequencies;
		let [a, b, c, d, e, f] = self.get_rates();

		let mut gtr = vec![
			vec![
				-a * p_c - b * p_g - c * p_t,
				a * p_c,
				b * p_g,
				c * p_t,
			],
			vec![
				a * p_a,
				-a * p_a - d * p_g - e * p_t,
				d * p_g,
				e * p_t,
			],
			vec![
				b * p_a,
				d * p_c,
				-b * p_a - d * p_c - f * p_t,
				f * p_t,
			],
			vec![
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

		#[expect(clippy::needless_range_loop)]
		for i in 0..4 {
			for j in 0..4 {
				self.q[i][j] = gtr[i][j];
			}
		}

		let mut eigen_sys = beast_eigen::DefaultEigenSystem::new(4);
		let decomposition =
			eigen_sys.decompose_matrix(&mut gtr).unwrap();

		self.p.as_flattened_mut()
			.copy_from_slice(decomposition.get_eigen_vectors());
		self.inv_p.as_flattened_mut().copy_from_slice(
			decomposition.get_inverse_eigen_vectors(),
		);
		self.diag.copy_from_slice(decomposition.get_eigen_values());
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
		if distance > 20.0 {
			transition_freqs(&self.cached_frequencies, dst);
			return;
		} else if distance < 1e-10 {
			transition_q(&self.q, distance, dst);
			return;
		}

		let diag: M4 =
			from_diagonal(&self.diag.map(|v| (v * distance).exp()));

		let out = cmul(&cmul(&self.p, &diag), &self.inv_p);
		dst.copy_from(out.as_mat());
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
			self.has_changed = false;
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
	diag: Vec<f64>,
	scratch: Vec<f64>,
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
			diag: vec![0.0; n * n],
			scratch: vec![0.0; n * n],
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
			diag: self.diag.clone(),
			scratch: self.scratch.clone(),
			has_changed: self.has_changed,
		}
	}

	#[expect(clippy::needless_range_loop)]
	fn update_matrices(&mut self) {
		let n = self.eigenvalues.len();
		let freqs = self.frequencies.get().inner();
		let rates = self.rates.get().inner();

		let mut q = vec![vec![0.0; n]; n];

		let mut rate_idx = 0;
		// upper triangular
		for i in 0..n {
			for j in (i + 1)..n {
				q[i][j] = freqs[j] * rates[rate_idx];
				rate_idx += 1;
			}
		}

		let mut rate_idx = 0;
		// lower triangular
		for i in 0..n {
			for j in 0..i {
				q[i][j] = freqs[j] * rates[rate_idx];
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
				sum += q[i][j];
			}
			q[i][i] = -sum;
			scale += sum * freqs[i];
		}

		for row in &mut q {
			for el in row {
				*el /= scale;
			}
		}

		let mut eigen_sys =
			beast_eigen::DefaultEigenSystem::new(self.diag.len());
		let decomposition = eigen_sys.decompose_matrix(&mut q).unwrap();

		self.p.copy_from_slice(decomposition.get_eigen_vectors());
		self.inv_p.copy_from_slice(
			decomposition.get_inverse_eigen_vectors(),
		);
		self.diag.copy_from_slice(decomposition.get_eigen_values());
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

	fn write_transition(&mut self, distance: f64, dst: MatrixMut<'_, f64>) {
		let n = self.eigenvalues.len();
		assert!((*dst).is_square());
		assert_eq!((*dst).num_cols(), n);

		let mut scratch_ref = self.scratch.as_mat_mut(n, n);

		let p_ref = self.p.as_mat(n, n);
		let inv_p_ref = self.inv_p.as_mat(n, n);

		let mut diag_ref = self.diag.as_mat_mut(n, n);
		for i in 0..n {
			diag_ref[(i, i)] =
				(self.eigenvalues[i] * distance).exp();
		}

		mul(p_ref, *diag_ref, scratch_ref.reborrow());
		mul(*scratch_ref, inv_p_ref, dst);
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
