use anyhow::Result;
use log::trace;
use parking_lot::Mutex;
use pyo3::prelude::*;

use std::{collections::HashMap, slice};

use crate::{
	Transitions, clock::PyClock, substitution::BoxedSubstitutionModel,
	tree::PyTree,
};
use data::{DnaNucleotide, Msa, PyMsa, seq::Character};
use linalg::{RowMatrix, Vector};
use util::{py_call_method, py_check_method};

mod cpu;
mod cuda;
mod thread;

use cpu::CpuLikelihood;
use cuda::CudaLikelihood;
use thread::ThreadedLikelihood;

pub type Row<const N: usize> = Vector<f64, N>;
type Transition<const N: usize> = RowMatrix<f64, N, N>;

pub trait LikelihoodTrait<const N: usize> {
	type Arguments;

	fn new(msa: Msa<DnaNucleotide>, args: Self::Arguments) -> Result<Self>
	where
		Self: Sized;

	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<N>],
		leaves_end: usize,
		root: usize,
		frequencies: Vector<f64, N>,
	) -> Result<()>;

	fn likelihood(&mut self, weights: &[f64]) -> Result<f64>;

	fn accept(&mut self) -> Result<()>;

	fn reject(&mut self) -> Result<()>;
}

pub struct GenericLikelihood<const N: usize, L: LikelihoodTrait<N>> {
	transitions: Transitions<N>,
	calculator: L,
	weights: Vec<f64>,
	/// Last accepted likelihood
	cache: f64,
	/// Last calculated likelihood.  It's different from the cache, because
	/// it might get rejected.
	last: f64,
	launched_update: bool,
	tree: Py<PyTree>,
}

impl<L> GenericLikelihood<4, L>
where
	L: LikelihoodTrait<4>,
{
	fn new(
		substitution: BoxedSubstitutionModel<4>,
		clock: PyClock,
		msa: Msa<DnaNucleotide>,
		tree: Py<PyTree>,

		arguments: L::Arguments,
	) -> Result<Self> {
		let num_internals = msa.num_sequences() - 1;
		let transitions = Transitions::<4>::new(
			num_internals * 2,
			substitution,
			clock,
		);

		let (msa, weights) = deduplicate(msa);

		let calculator = L::new(msa, arguments)?;

		let mut out = Self {
			transitions,
			calculator,
			weights,
			cache: f64::NAN,
			last: f64::NAN,
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

impl<const N: usize, L: LikelihoodTrait<N>> GenericLikelihood<N, L> {
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
			full_update;
			""
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

		let likelihood = self.calculator.likelihood(&self.weights)?;
		trace!(
			target: "b3::likelihood::likelihood",
			likelihood;
			""
		);
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

#[pyclass(name = "CPU4Likelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct PyCpu4Likelihood {
	inner: Mutex<GenericLikelihood<4, CpuLikelihood<4>>>,
}

#[pymethods]
impl PyCpu4Likelihood {
	#[new]
	fn new(
		msa: PyMsa,
		substitution: BoxedSubstitutionModel<4>,
		clock: PyClock,
		tree: Py<PyTree>,
	) -> Result<Self> {
		let generic = GenericLikelihood::new(
			substitution,
			clock,
			msa.0,
			tree,
			(),
		)?;

		Ok(Self {
			inner: Mutex::new(generic),
		})
	}
}

likelihood_methods!(PyCpu4Likelihood);

#[pyclass(
	name = "Thread4Likelihood",
	module = "aspartik.b3.likelihoods",
	frozen
)]
pub struct PyThread4Likelihood {
	inner: Mutex<GenericLikelihood<4, ThreadedLikelihood<4>>>,
}

#[pymethods]
impl PyThread4Likelihood {
	#[new]
	#[pyo3(signature = (
		msa, substitution, clock, tree,
		*,
		thread_split_size = 400,
	))]
	fn new(
		msa: PyMsa,
		substitution: BoxedSubstitutionModel<4>,
		clock: PyClock,
		tree: Py<PyTree>,
		thread_split_size: usize,
	) -> Result<Self> {
		let generic = GenericLikelihood::new(
			substitution,
			clock,
			msa.0,
			tree,
			(thread_split_size,),
		)?;

		Ok(Self {
			inner: Mutex::new(generic),
		})
	}
}

likelihood_methods!(PyThread4Likelihood);

#[pyclass(name = "CUDALikelihood", module = "aspartik.b3.likelihoods", frozen)]
pub struct PyCudaLikelihood {
	inner: Mutex<GenericLikelihood<4, CudaLikelihood>>,
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
		substitution: BoxedSubstitutionModel<4>,
		clock: PyClock,
		tree: Py<PyTree>,
		cuda_device: usize,
	) -> Result<Self> {
		let generic = GenericLikelihood::new(
			substitution,
			clock,
			msa.0,
			tree,
			(cuda_device,),
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
