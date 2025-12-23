use anyhow::Result;
use num_traits::{Float, NumAssignOps, NumCast};
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::{
	collections::HashMap,
	fmt::Debug,
	ops::{DivAssign, Mul, MulAssign},
	slice,
};

use crate::{
	Transitions,
	clock::PyClock,
	substitution::{BoxedSubstitutionModel, Substitution4},
	tree::PyTree,
};
use data::{DnaNucleotide, Msa, PyMsa, seq::Character};
use linalg::{RowMatrix, Vector};
use logger::trace;
use util::{py_call_method, py_check_method};

mod cpu;
mod cuda;
mod parallel;

use cpu::CpuLikelihood;
use cuda::CudaLikelihood;
use parallel::ParallelLikelihood;

pub trait Space {
	type Scalar: Float
		+ NumAssignOps
		+ NumCast
		+ From<f64>
		+ Sync
		+ Send
		+ Debug
		+ 'static;
	type Vector: Mul<Output = Self::Vector>
		+ MulAssign<Self::Scalar>
		+ DivAssign<Self::Scalar>
		+ PartialOrd<Self::Scalar>
		+ DivAssign
		+ Copy
		+ Sync
		+ Send
		+ Debug
		+ 'static;
	type Matrix: Mul<Self::Vector, Output = Self::Vector>
		+ Copy
		+ Default
		+ Sync
		+ Send
		+ Debug
		+ 'static;

	fn sum(v: Self::Vector) -> Self::Scalar;
}

pub struct Linalg4;
impl Space for Linalg4 {
	type Scalar = f64;
	type Vector = Vector<f64, 4>;
	type Matrix = RowMatrix<f64, 4, 4>;

	fn sum(v: Self::Vector) -> Self::Scalar {
		v.sum()
	}
}

pub trait LikelihoodTrait {
	type S: Space;

	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[<Self::S as Space>::Matrix],
		leaves_end: usize,
		root: usize,
		frequencies: <Self::S as Space>::Vector,
	) -> Result<()>;

	fn likelihood(&mut self) -> Result<<Self::S as Space>::Scalar>;

	fn accept(&mut self) -> Result<()>;

	fn reject(&mut self) -> Result<()>;
}

pub struct GenericLikelihood<S, L>
where
	S: Space,
	L: LikelihoodTrait<S = S>,
{
	calculator: L,
	transitions: Transitions<S>,
	/// Last accepted likelihood
	cache: S::Scalar,
	/// Last calculated likelihood.  It's different from the cache, because
	/// it might get rejected.
	last: S::Scalar,
	launched_update: bool,
	tree: Py<PyTree>,
}

impl<S, L> GenericLikelihood<S, L>
where
	S: Space,
	L: LikelihoodTrait<S = S>,
{
	fn new(
		calculator: L,
		substitution: BoxedSubstitutionModel<S>,
		clock: PyClock,
		tree: Py<PyTree>,
	) -> Result<Self> {
		let transitions = Transitions::new(
			tree.get().num_edges(),
			substitution,
			clock,
		);

		let mut out = Self {
			calculator,
			transitions,
			cache: f64::NAN.into(),
			last: f64::NAN.into(),
			launched_update: false,
			tree,
		};
		Python::attach(|py| out.propose(py))?;
		// This cannot be removed: the likelihood must be run to
		// completion in case the calculator is async.
		out.likelihood()?;
		// propose sets `last` and accept updates the cache, so neither
		// cache nor last will be NaN.
		out.accept()?;
		Ok(out)
	}

	fn propose(&mut self, py: Python) -> Result<()> {
		let tree = &mut self.tree.get().inner();
		let full_update = self.transitions.update(py, tree)?;
		let (nodes, leaves_end) = if full_update {
			tree.full_update()
		} else {
			tree.nodes_to_update()
		};
		trace!(
			target: "b3::likelihood::propose",
			num_nodes_to_update = nodes.len(),
			full_update = full_update
		);

		// no tree update, return the cache
		if nodes.is_empty() {
			self.launched_update = false;
			return Ok(());
		}

		let (nodes, edges, root) = tree.to_lists(&nodes);

		let transitions = self.transitions.matrices(&edges);

		let frequencies = self.transitions.frequencies();

		self.calculator.propose(
			&nodes,
			&edges,
			&transitions,
			leaves_end,
			root,
			frequencies,
		)?;
		self.launched_update = true;

		Ok(())
	}

	fn likelihood(&mut self) -> Result<S::Scalar> {
		if !self.launched_update {
			self.last = self.cache;
			return Ok(self.cache);
		}

		let likelihood = self.calculator.likelihood()?;
		self.last = likelihood;
		Ok(likelihood)
	}

	fn accept(&mut self) -> Result<()> {
		self.cache = self.last;
		self.launched_update = false;
		self.calculator.accept()?;
		self.transitions.accept();
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.launched_update = false;
		self.calculator.reject()?;
		self.transitions.reject();
		Ok(())
	}
}

fn deduplicate(mut msa: Msa<DnaNucleotide>) -> (Msa<DnaNucleotide>, Vec<f64>) {
	let mut hashes =
		Vec::<(usize, blake3::Hash)>::with_capacity(msa.num_sites());

	let mut hasher = blake3::Hasher::new();
	for site in 0..msa.num_sites() {
		for seq in 0..msa.num_sequences() {
			let byte = msa.sequence(seq)[site].to_byte();
			hasher.update(slice::from_ref(&byte));
		}

		hashes.push((site, hasher.finalize()));
		hasher.reset();
	}

	// hash -> (index, count)
	let mut map = HashMap::<blake3::Hash, (usize, f64)>::new();

	for (index, hash) in &hashes {
		if let Some((_, count)) = map.get_mut(hash) {
			// there's an earlier site with the same contents
			*count += 1.0;
		} else {
			map.insert(*hash, (*index, 1.0));
		}
	}

	let mut pairs: Vec<_> = map.values().collect();
	pairs.sort_by_key(|(index, _)| index);

	let (indices, weights): (Vec<_>, Vec<_>) =
		pairs.iter().copied().copied().unzip();

	msa.set_sites(indices);

	(msa, weights)
}

macro_rules! likelihood_methods {
	($type:ty) => {
		#[pymethods]
		impl $type {
			fn propose(&self, py: Python) -> Result<()> {
				self.inner.lock().propose(py)
			}

			fn likelihood(&self) -> Result<f64> {
				self.inner.lock().likelihood()
			}

			fn accept(&self) -> Result<()> {
				self.inner.lock().accept()
			}

			fn reject(&self) -> Result<()> {
				self.inner.lock().reject()
			}
		}
	};
}

/// 4-state DNA likelihood calculator.
///
/// It's synchronous.  `CUDALikelihood` should be used for parallel calculations
/// for alingments larger than 100Kb.
#[pyclass(name = "CPU4Likelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct PyCpu4Likelihood {
	inner: Mutex<GenericLikelihood<Linalg4, CpuLikelihood<Linalg4>>>,
}

#[pymethods]
impl PyCpu4Likelihood {
	#[new]
	#[pyo3(signature = (msa, substitution, clock, tree, scale = 1e-40))]
	fn new(
		msa: PyMsa,
		substitution: Substitution4,
		clock: PyClock,
		tree: Py<PyTree>,
		scale: f64,
	) -> Result<Self> {
		let calculator = CpuLikelihood::new(msa.0, scale);
		let generic = GenericLikelihood::new(
			calculator,
			substitution,
			clock,
			tree,
		)?;

		Ok(Self {
			inner: Mutex::new(generic),
		})
	}
}

likelihood_methods!(PyCpu4Likelihood);

#[pyclass(
	name = "Parallel4Likelihood",
	module = "aspartik.b3.likelihoods",
	frozen
)]
pub struct PyParallel4Likelihood {
	inner: Mutex<GenericLikelihood<Linalg4, ParallelLikelihood<Linalg4>>>,
}

#[pymethods]
impl PyParallel4Likelihood {
	#[new]
	#[pyo3(signature = (msa, substitution, clock, tree, num_threads, scale_ln = 30))]
	fn new(
		msa: PyMsa,
		substitution: Substitution4,
		clock: PyClock,
		tree: Py<PyTree>,
		num_threads: usize,
		scale_ln: u32,
	) -> Result<Self> {
		let calculator =
			ParallelLikelihood::new(msa.0, num_threads, scale_ln)?;
		let generic = GenericLikelihood::new(
			calculator,
			substitution,
			clock,
			tree,
		)?;

		Ok(Self {
			inner: Mutex::new(generic),
		})
	}
}

likelihood_methods!(PyParallel4Likelihood);

/// Likelihood calculations on NVIDIA graphics cards.
///
/// Only supports 4-state DNA models.  `cuda_device` allows selecting the device
/// index.
#[pyclass(name = "CUDALikelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct PyCudaLikelihood {
	inner: Mutex<GenericLikelihood<Linalg4, CudaLikelihood>>,
}

#[pymethods]
impl PyCudaLikelihood {
	#[new]
	#[pyo3(signature = (
		msa, substitution, clock, tree,
		*,
		cuda_device= 0,
	))]
	fn new(
		msa: PyMsa,
		substitution: Substitution4,
		clock: PyClock,
		tree: Py<PyTree>,
		cuda_device: usize,
	) -> Result<Self> {
		let calculator = CudaLikelihood::new(msa.0, cuda_device)?;
		let generic = GenericLikelihood::new(
			calculator,
			substitution,
			clock,
			tree,
		)?;

		Ok(Self {
			inner: Mutex::new(generic),
		})
	}

	fn pattern_likelihoods(&self) -> Result<Vec<f64>> {
		let generic = &mut *self.inner.lock();
		let root = generic.tree.get().root();
		let frequencies = generic.transitions.frequencies();

		let likelihoods = generic
			.calculator
			.pattern_likelihoods(root, frequencies)?;

		Ok(likelihoods)
	}
}

likelihood_methods!(PyCudaLikelihood);

pub struct PyLikelihood {
	inner: Py<PyAny>,
}

impl<'py> FromPyObject<'_, 'py> for PyLikelihood {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "propose");
		py_check_method!(obj, "likelihood");
		py_check_method!(obj, "accept");
		py_check_method!(obj, "reject");

		Ok(PyLikelihood {
			inner: obj.to_owned().unbind(),
		})
	}
}

impl PyLikelihood {
	pub fn propose(&self, py: Python) -> Result<()> {
		py_call_method!(py, self.inner, "propose")?;
		Ok(())
	}

	pub fn likelihood(&self, py: Python) -> Result<f64> {
		let out = py_call_method!(py, self.inner, "likelihood")?
			.extract(py)?;
		Ok(out)
	}

	pub fn accept(&self, py: Python) -> Result<()> {
		py_call_method!(py, self.inner, "accept")?;
		Ok(())
	}

	pub fn reject(&self, py: Python) -> Result<()> {
		py_call_method!(py, self.inner, "reject")?;
		Ok(())
	}

	pub fn clone_ref(&self, py: Python) -> Py<PyAny> {
		self.inner.clone_ref(py)
	}
}
