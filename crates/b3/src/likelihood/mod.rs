#![expect(unused)]

use anyhow::Result;
use num_traits::Float;
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::{collections::HashMap, fmt::Debug, slice};

use crate::{
	Transitions,
	clock::PyClock,
	parameters::PyTree,
	substitution::{PySubstitution4, SubstitutionModel},
};
use data::{DnaNucleotide, Msa, PyMsa, seq::Character};
use logger::{info, trace};

mod cpu;
mod cuda;
mod hetero;
mod parallel;

use cpu::CpuLikelihood;
use cuda::CudaLikelihood;
pub use hetero::PyHeteroLikelihood;
use parallel::ParallelLikelihood;

pub trait LikelihoodTrait<const N: usize, F> {
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[(usize, usize)],
		transitions: &[[[F; N]; N]],
		leaves_end: usize,
		frequencies: [F; N],
	) -> Result<()>;

	fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()>;

	fn accept(&mut self) -> Result<()>;

	fn reject(&mut self) -> Result<()>;
}

pub struct GenericLikelihood<const N: usize, F, L, S> {
	calculator: L,
	pattern_likelihoods: Vec<f64>,
	backup_pattern_likelihoods: Vec<f64>,
	pattern_weights: Vec<u32>,

	transitions: Transitions<N, F, S>,
	tree: Py<PyTree>,

	/// Last accepted likelihood
	cache: f64,
	/// Last calculated likelihood.  It's different from the cache, because
	/// it might get rejected.
	last: f64,
	launched_update: bool,
}

impl<const N: usize, F, L, S> GenericLikelihood<N, F, L, S>
where
	F: Float + Default,
	L: LikelihoodTrait<N, F>,
	S: SubstitutionModel<N, F>,
{
	fn new(
		calculator: L,
		weights: Vec<u32>,
		substitution: S,
		clock: Py<PyClock>,
		tree: Py<PyTree>,
	) -> Result<Self> {
		info!(
			target: "b3::likelihood::GenericLikelihood::new",
			weights_len = weights.len()
		);

		let transitions = Transitions::new(
			tree.get().num_nodes(),
			substitution,
			clock,
		);

		let mut out = Self {
			calculator,
			pattern_likelihoods: vec![f64::NAN; weights.len()],
			backup_pattern_likelihoods: vec![
				f64::NAN;
				weights.len()
			],
			pattern_weights: weights,

			transitions,
			tree,

			cache: f64::NAN,
			last: f64::NAN,
			launched_update: false,
		};
		out.propose()?;
		// This cannot be removed: the likelihood must be run to
		// completion in case the calculator is async.
		out.likelihood()?;
		// propose sets `last` and accept updates the cache, so neither
		// cache nor last will be NaN.
		out.accept()?;
		Ok(out)
	}

	fn propose(&mut self) -> Result<()> {
		let tree = &mut self.tree.get().inner();
		self.transitions.update(tree)?;
		let (nodes, leaves_end) = tree.nodes_to_update();

		trace!(
			target: "b3::likelihood::propose",
			num_nodes_to_update = nodes.len()
		);

		// no tree update, return the cache
		if nodes.is_empty() {
			self.launched_update = false;
			return Ok(());
		}

		let (nodes, children) = tree.to_lists(&nodes);

		let transitions =
			self.transitions.matrices(&nodes[..nodes.len() - 1]);

		let frequencies = self.transitions.frequencies();

		assert_eq!(nodes.len() - leaves_end, children.len());
		assert_eq!(nodes.len() - 1, transitions.len());

		self.calculator.propose(
			&nodes,
			&children,
			&transitions,
			leaves_end,
			frequencies,
		)?;
		self.launched_update = true;

		Ok(())
	}

	fn likelihood(&mut self) -> Result<f64> {
		if !self.launched_update {
			self.last = self.cache;
			return Ok(self.cache);
		}

		self.calculator.likelihood(&mut self.pattern_likelihoods)?;

		for (likelihood, weight) in self
			.pattern_likelihoods
			.iter_mut()
			.zip(&self.pattern_weights)
		{
			*likelihood *= f64::from(*weight);
		}

		self.last = self.pattern_likelihoods.iter().sum();
		Ok(self.last)
	}

	fn accept(&mut self) -> Result<()> {
		self.cache = self.last;
		if self.launched_update {
			self.calculator.accept()?;
			self.transitions.accept();
			self.backup_pattern_likelihoods
				.copy_from_slice(&self.pattern_likelihoods);
		}
		self.launched_update = false;
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		if self.launched_update {
			self.calculator.reject()?;
			self.transitions.reject();
			self.pattern_likelihoods.copy_from_slice(
				&self.backup_pattern_likelihoods,
			);
		}
		self.launched_update = false;
		Ok(())
	}

	fn num_patterns(&self) -> usize {
		self.pattern_weights.len()
	}

	fn pattern_likelihoods(&self) -> Result<Vec<f64>> {
		Ok(self.pattern_likelihoods.clone())
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
	($type:ty) => {
		#[pymethods]
		impl $type {
			fn propose(&self) -> Result<()> {
				self.inner.lock().propose()
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

		impl $type {
			pub fn pattern_likelihoods(&self) -> Result<Vec<f64>> {
				self.inner.lock().pattern_likelihoods()
			}

			pub fn num_patterns(&self) -> usize {
				self.inner.lock().num_patterns()
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
	inner: Mutex<
		GenericLikelihood<
			4,
			f64,
			CpuLikelihood<4, f64>,
			PySubstitution4,
		>,
	>,
}

#[pymethods]
impl PyCpu4Likelihood {
	#[new]
	#[pyo3(signature = (msa, substitution, clock, tree, scale_ln = 30))]
	fn new(
		msa: Py<PyMsa>,
		substitution: PySubstitution4,
		clock: Py<PyClock>,
		tree: Py<PyTree>,
		scale_ln: u32,
	) -> Result<Self> {
		let (leaves, weights) = deduplicate(&msa.get().inner());
		let calculator =
			CpuLikelihood::new(weights.len(), leaves, scale_ln);
		let generic = GenericLikelihood::new(
			calculator,
			weights,
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
	inner: Mutex<
		GenericLikelihood<
			4,
			f64,
			ParallelLikelihood<4, f64>,
			PySubstitution4,
		>,
	>,
}

#[pymethods]
impl PyParallel4Likelihood {
	#[new]
	#[pyo3(signature = (
		msa, substitution, clock, tree,
		num_leaf_threads = 0, num_internal_threads = 3, scale_ln = 30
	))]
	fn new(
		msa: Py<PyMsa>,
		substitution: PySubstitution4,
		clock: Py<PyClock>,
		tree: Py<PyTree>,
		mut num_leaf_threads: usize,
		num_internal_threads: usize,
		scale_ln: u32,
	) -> Result<Self> {
		if num_leaf_threads == 0 {
			num_leaf_threads = num_internal_threads;
		}

		let (leaves, weights) = deduplicate(&msa.get().inner());
		let calculator = ParallelLikelihood::new(
			weights.len(),
			leaves,
			num_leaf_threads,
			num_internal_threads,
			scale_ln,
		)?;
		let generic = GenericLikelihood::new(
			calculator,
			weights,
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
	inner: Mutex<
		GenericLikelihood<4, f64, CudaLikelihood, PySubstitution4>,
	>,
}

#[pymethods]
impl PyCudaLikelihood {
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
		let (leaves, weights) = deduplicate(&msa.get().inner());
		let calculator = CudaLikelihood::new(
			weights.len(),
			leaves,
			scale_ln,
			cuda_device,
		)?;
		let generic = GenericLikelihood::new(
			calculator,
			weights,
			substitution,
			clock,
			tree,
		)?;

		Ok(Self {
			inner: Mutex::new(generic),
		})
	}
}

likelihood_methods!(PyCudaLikelihood);

pub enum PyLikelihood {
	Cpu(Py<PyCpu4Likelihood>),
	Parallel(Py<PyParallel4Likelihood>),
	Cuda(Py<PyCudaLikelihood>),
	Hetero(Py<PyHeteroLikelihood>),
}

impl<'py> FromPyObject<'_, 'py> for PyLikelihood {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		if let Ok(l) = obj.cast::<PyCpu4Likelihood>() {
			Ok(Self::Cpu(l.into()))
		} else if let Ok(l) = obj.cast::<PyParallel4Likelihood>() {
			Ok(Self::Parallel(l.into()))
		} else if let Ok(l) = obj.cast::<PyCudaLikelihood>() {
			Ok(Self::Cuda(l.into()))
		} else if let Ok(l) = obj.cast::<PyHeteroLikelihood>() {
			Ok(Self::Hetero(l.into()))
		} else {
			todo!("descriptive error")
		}
	}
}

impl<'py> IntoPyObject<'py> for PyLikelihood {
	type Target = PyAny;
	type Output = Bound<'py, PyAny>;
	type Error = PyErr;

	fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, PyErr> {
		Ok(match self {
			Self::Cpu(l) => {
				Bound::new(py, l.clone_ref(py))?.into_any()
			}
			Self::Parallel(l) => {
				Bound::new(py, l.clone_ref(py))?.into_any()
			}
			Self::Cuda(l) => {
				Bound::new(py, l.clone_ref(py))?.into_any()
			}
			Self::Hetero(l) => {
				Bound::new(py, l.clone_ref(py))?.into_any()
			}
		})
	}
}

impl PyLikelihood {
	pub fn clone_ref(&self, py: Python) -> Self {
		match self {
			Self::Cpu(l) => Self::Cpu(l.clone_ref(py)),
			Self::Parallel(l) => Self::Parallel(l.clone_ref(py)),
			Self::Cuda(l) => Self::Cuda(l.clone_ref(py)),
			Self::Hetero(l) => Self::Hetero(l.clone_ref(py)),
		}
	}

	pub fn propose(&self) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().propose(),
			Self::Parallel(l) => l.get().propose(),
			Self::Cuda(l) => l.get().propose(),
			Self::Hetero(l) => l.get().propose(),
		}
	}

	pub fn likelihood(&self) -> Result<f64> {
		match self {
			Self::Cpu(l) => l.get().likelihood(),
			Self::Parallel(l) => l.get().likelihood(),
			Self::Cuda(l) => l.get().likelihood(),
			Self::Hetero(l) => l.get().likelihood(),
		}
	}

	pub fn accept(&self) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().accept(),
			Self::Parallel(l) => l.get().accept(),
			Self::Cuda(l) => l.get().accept(),
			Self::Hetero(l) => l.get().accept(),
		}
	}

	pub fn reject(&self) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().reject(),
			Self::Parallel(l) => l.get().reject(),
			Self::Cuda(l) => l.get().reject(),
			Self::Hetero(l) => l.get().reject(),
		}
	}

	pub fn num_patterns(&self) -> usize {
		match self {
			Self::Cpu(l) => l.get().num_patterns(),
			Self::Parallel(l) => l.get().num_patterns(),
			Self::Cuda(l) => l.get().num_patterns(),
			Self::Hetero(_l) => todo!(),
		}
	}

	pub fn pattern_likelihoods(&self) -> Result<Vec<f64>> {
		match self {
			Self::Cpu(l) => l.get().pattern_likelihoods(),
			Self::Parallel(l) => l.get().pattern_likelihoods(),
			Self::Cuda(l) => l.get().pattern_likelihoods(),
			Self::Hetero(_l) => todo!(),
		}
	}
}
