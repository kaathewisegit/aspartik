use anyhow::Result;
use num_traits::Float;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;

use std::{collections::HashMap, slice};

use crate::{
	Transitions,
	clock::PyClock,
	parameters::{Parameter, PyTree, Tree},
	substitution::{PySubstitution4, SubstitutionModel},
};
use data::{DnaNucleotide, Msa, PyMsa, seq::Character};

mod cpu;
mod cuda;
mod hetero;

use cpu::Cpu4Propagations;
use cuda::CudaLikelihood;
pub use hetero::PyHeteroLikelihood;

/// Felsenstein's pruning tree likelihood calculator
///
/// This trait is for low-level implementations of Felsenstein's likelihood
/// algorithm which are used by `GenericLikelihood`.  The latter takes care of
/// the substitution model and the tree, so a calculator only needs to implement
/// the raw calculations.
///
/// The trait is generic over `N`, the dimensionality of data, and `F`, the
/// floating point type.  For now all likelihoods use `N = 4` (DNA nucleotides)
/// and `f64`.
///
/// Calculators have a simple life cycle.  On a step where either the tree, the
/// clock rate, or the substitution models are edited, `GenericLikelihood`
/// figures out the minimal set of nodes to update, the transition matrices
/// relevant for the update, and calls `propose`.  After that it calls
/// `likelihood`, and, finally, either `accept` or `reject`.
///
/// `propose` and `likelihood` are split into two different methods to allow
/// asynchronous implementations.  Some higher-level likelihoods compose over
/// several calculators.  In this case, they'll first call `propose` on each
/// calculator and then block on `likelihood` calls.
pub trait Calculator<const N: usize, F> {
	/// Calculate tree likelihood
	fn likelihood(
		&mut self,
		tree: MutexGuard<Tree>,
		transitions: &Transitions<N, F>,
	) -> Result<f64>;

	/// Accept the changes made in `likelihood`
	fn accept(&mut self) -> Result<()>;

	/// Reject the changes made in `likelihood`
	///
	/// This should roll back the internal state of the calculator to
	/// exactly what it was after the last call to `accept`.
	fn reject(&mut self) -> Result<()>;

	/// Number of patterns in the alignment
	fn num_patterns(&self) -> usize;
}

pub struct GenericLikelihood<const N: usize, F, L> {
	calculator: L,

	transitions: Transitions<N, F>,
	tree: Py<PyTree>,

	/// Last accepted likelihood
	cache: f64,
	/// Last calculated likelihood.  It's different from the cache, because
	/// it might get rejected.
	last: f64,
	launched_update: bool,
}

impl<const N: usize, F, L> GenericLikelihood<N, F, L>
where
	F: Float + Default,
	L: Calculator<N, F>,
{
	fn new<S>(
		calculator: L,
		substitution: S,
		clock: Py<PyClock>,
		tree: Py<PyTree>,
	) -> Result<Self>
	where
		S: SubstitutionModel<N, F> + Sync + Send + 'static,
	{
		let transitions = Transitions::new(
			tree.get().num_nodes(),
			substitution,
			clock,
		);

		let mut out = Self {
			calculator,

			transitions,
			tree,

			cache: f64::NAN,
			last: f64::NAN,
			launched_update: false,
		};
		// This cannot be removed: the likelihood must be run to
		// completion in case the calculator is async.
		out.likelihood()?;
		// likelihood sets `last` and accept updates the cache, so
		// neither cache nor last will be NaN.
		out.accept()?;
		Ok(out)
	}

	fn likelihood(&mut self) -> Result<f64> {
		let mut tree = self.tree.get().inner();
		self.transitions.update(&mut tree)?;

		// no tree update, return the cache
		if !tree.is_changed() {
			self.launched_update = false;
			self.last = self.cache;
			return Ok(self.cache);
		}

		self.launched_update = true;
		self.last =
			self.calculator.likelihood(tree, &self.transitions)?;
		Ok(self.last)
	}

	fn accept(&mut self) -> Result<()> {
		self.cache = self.last;
		if self.launched_update {
			self.calculator.accept()?;
			self.transitions.accept();
		}
		self.launched_update = false;
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		if self.launched_update {
			self.calculator.reject()?;
			self.transitions.reject();
		}
		self.launched_update = false;
		Ok(())
	}

	fn num_patterns(&self) -> usize {
		self.calculator.num_patterns()
	}
}

fn deduplicate(msa: &Msa<DnaNucleotide>) -> (Vec<u8>, Vec<u32>) {
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
	let mut map = HashMap::<blake3::Hash, (usize, u32)>::new();

	for (index, hash) in &hashes {
		if let Some((_, count)) = map.get_mut(hash) {
			// there's an earlier site with the same contents
			*count += 1;
		} else {
			map.insert(*hash, (*index, 1));
		}
	}

	let mut pairs: Vec<_> = map.values().collect();
	pairs.sort_by_key(|(index, _)| index);

	let (indices, weights): (Vec<_>, Vec<_>) =
		pairs.iter().copied().copied().unzip();

	let mut leaves =
		Vec::with_capacity(msa.num_sequences() * indices.len());

	fn char_to_u8(ch: DnaNucleotide) -> u8 {
		match ch {
			DnaNucleotide::Gap => 0b1111,
			ch => ch.to_byte(),
		}
	}

	for seq in 0..msa.num_sequences() {
		for site in indices.iter().copied() {
			let char = msa.sequence(seq)[site];
			leaves.push(char_to_u8(char))
		}
	}

	(leaves, weights)
}

macro_rules! likelihood_methods {
	($type:ty $(; $($rest:tt)*)?) => {
		#[pymethods]
		impl $type {
			fn likelihood(&self) -> Result<f64> {
				self.inner.lock().likelihood()
			}

			fn accept(&self) -> Result<()> {
				self.inner.lock().accept()
			}

			fn reject(&self) -> Result<()> {
				self.inner.lock().reject()
			}

			pub fn num_patterns(&self) -> usize {
				self.inner.lock().num_patterns()
			}

			$($($rest)*)?
		}

		impl $type {
			pub fn pattern_likelihoods(&self) -> Result<Vec<f64>> {
				todo!()
			}
		}
	};
}

/// 4-state DNA likelihood calculator.
///
/// It's synchronous.  `CUDALikelihood` should be used for parallel calculations
/// for alignments larger than 100Kb.
#[pyclass(name = "CPU4Likelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct PyCpu4Likelihood {
	inner: Mutex<GenericLikelihood<4, f64, Cpu4Propagations>>,
}

likelihood_methods! {PyCpu4Likelihood;
	#[new]
	#[pyo3(signature = (
		msa, substitution, clock, tree,
		num_threads = 0, scale_ln = 30
	))]
	fn new(
		msa: Py<PyMsa>,
		substitution: PySubstitution4,
		clock: Py<PyClock>,
		tree: Py<PyTree>,
		num_threads: usize,
		scale_ln: u32,
	) -> Result<Self> {
		let (leaves, weights) = deduplicate(msa.get());
		let calculator = Cpu4Propagations::new(
			weights,
			leaves,
			num_threads,
			scale_ln,
		);
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

/// Likelihood calculations on NVIDIA graphics cards.
///
/// Only supports 4-state DNA models.  `cuda_device` allows selecting the device
/// index.
#[pyclass(name = "CUDALikelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct PyCudaLikelihood {
	inner: Mutex<GenericLikelihood<4, f64, CudaLikelihood>>,
}

likelihood_methods! {PyCudaLikelihood;
	#[new]
	#[pyo3(signature = (
		msa, substitution, clock, tree,
		*,
		scale_ln = 30,
		cuda_device= 0,
	))]
	fn new(
		msa: Py<PyMsa>,
		substitution: PySubstitution4,
		clock: Py<PyClock>,
		tree: Py<PyTree>,
		scale_ln: u32,
		cuda_device: usize,
	) -> Result<Self> {
		let (leaves, weights) = deduplicate(msa.get());
		let calculator = CudaLikelihood::new(
			weights,
			leaves,
			scale_ln,
			cuda_device,
		)?;
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

#[derive(FromPyObject, IntoPyObject)]
pub enum PyLikelihood {
	Cpu(Py<PyCpu4Likelihood>),
	Cuda(Py<PyCudaLikelihood>),
	Hetero(Py<PyHeteroLikelihood>),
}

impl PyLikelihood {
	pub fn clone_ref(&self, py: Python) -> Self {
		match self {
			Self::Cpu(l) => Self::Cpu(l.clone_ref(py)),
			Self::Cuda(l) => Self::Cuda(l.clone_ref(py)),
			Self::Hetero(l) => Self::Hetero(l.clone_ref(py)),
		}
	}

	pub fn likelihood(&self) -> Result<f64> {
		match self {
			Self::Cpu(l) => l.get().likelihood(),
			Self::Cuda(l) => l.get().likelihood(),
			Self::Hetero(l) => l.get().likelihood(),
		}
	}

	pub fn accept(&self) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().accept(),
			Self::Cuda(l) => l.get().accept(),
			Self::Hetero(l) => l.get().accept(),
		}
	}

	pub fn reject(&self) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().reject(),
			Self::Cuda(l) => l.get().reject(),
			Self::Hetero(l) => l.get().reject(),
		}
	}

	pub fn num_patterns(&self) -> usize {
		match self {
			Self::Cpu(l) => l.get().num_patterns(),
			Self::Cuda(l) => l.get().num_patterns(),
			Self::Hetero(l) => l.get().num_patterns(),
		}
	}

	pub fn pattern_likelihoods(&self) -> Result<Vec<f64>> {
		match self {
			Self::Cpu(l) => l.get().pattern_likelihoods(),
			Self::Cuda(l) => l.get().pattern_likelihoods(),
			Self::Hetero(_l) => todo!(),
		}
	}
}
