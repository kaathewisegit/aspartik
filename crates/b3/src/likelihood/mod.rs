use anyhow::Result;
use num_traits::Float;
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::{collections::HashMap, fmt::Debug, slice};

use crate::{
	Transitions,
	clock::PyClock,
	parameters::PyTree,
	substitution::{BoxedSubstitutionModel, Substitution4},
};
use data::{DnaNucleotide, Msa, PyMsa, seq::Character};
use logger::{info, trace};
use util::{py_call_method, py_check_method};

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
		edges: &[usize],
		transitions: &[[[F; N]; N]],
		leaves_end: usize,
		root: usize,
		frequencies: [F; N],
	) -> Result<()>;

	fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()>;

	fn accept(&mut self) -> Result<()>;

	fn reject(&mut self) -> Result<()>;
}

pub struct GenericLikelihood<const N: usize, F, L> {
	calculator: L,
	pattern_likelihoods: Vec<f64>,
	pattern_weights: Vec<u32>,

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
	L: LikelihoodTrait<N, F>,
{
	fn new(
		calculator: L,
		weights: Vec<u32>,
		substitution: BoxedSubstitutionModel<N, F>,
		clock: Py<PyClock>,
		tree: Py<PyTree>,
	) -> Result<Self> {
		info!(
			target: "b3::likelihood::GenericLikelihood::new",
			weights_len = weights.len()
		);

		let transitions = Transitions::new(
			tree.get().num_edges(),
			substitution,
			clock,
		);

		let mut out = Self {
			calculator,
			pattern_likelihoods: vec![f64::NAN; weights.len()],
			pattern_weights: weights,

			transitions,
			tree,

			cache: f64::NAN,
			last: f64::NAN,
			launched_update: false,
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
		self.transitions.update(py, tree)?;
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

	fn likelihood(&mut self) -> Result<f64> {
		if !self.launched_update {
			self.last = self.cache;
			return Ok(self.cache);
		}

		self.calculator.likelihood(&mut self.pattern_likelihoods)?;

		let mut total_likelihood = 0.0;
		for (likelihood, weight) in self
			.pattern_likelihoods
			.iter()
			.zip(&self.pattern_weights)
		{
			total_likelihood += *likelihood * f64::from(*weight);
		}

		self.last = total_likelihood;
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
		self.pattern_weights.len()
	}

	fn pattern_likelihoods(&mut self) -> Result<Vec<f64>> {
		let mut out = vec![0.0; self.num_patterns()];

		self.calculator.likelihood(&mut out)?;

		for (likelihood, weight) in
			out.iter_mut().zip(&self.pattern_weights)
		{
			*likelihood *= f64::from(*weight);
		}

		Ok(out)
	}
}

fn deduplicate(msa: &Msa<DnaNucleotide>) -> (Vec<[f64; 4]>, Vec<u32>) {
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

	for seq in 0..msa.num_sequences() {
		for site in indices.iter().copied() {
			let char = msa.sequence(seq)[site];
			leaves.push(char.base_frequencies_denormalized())
		}
	}

	(leaves, weights)
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
/// for alingments larger than 100Kb.
#[pyclass(name = "CPU4Likelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct PyCpu4Likelihood {
	inner: Mutex<GenericLikelihood<4, f64, CpuLikelihood<4, f64>>>,
}

#[pymethods]
impl PyCpu4Likelihood {
	#[new]
	#[pyo3(signature = (msa, substitution, clock, tree, scale_ln = 30))]
	fn new(
		msa: Py<PyMsa>,
		substitution: Substitution4,
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
	inner: Mutex<GenericLikelihood<4, f64, ParallelLikelihood<4, f64>>>,
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
		substitution: Substitution4,
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
	inner: Mutex<GenericLikelihood<4, f64, CudaLikelihood>>,
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
		substitution: Substitution4,
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
