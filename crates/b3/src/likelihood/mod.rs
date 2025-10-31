use anyhow::{Result, bail};
use log::trace;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;

use std::{collections::HashMap, slice};

use crate::{
	Transitions, clock::PyClock, substitution::BoxedSubstitutionModel,
	tree::PyTree,
};
use data::{DnaNucleotide, Msa, PyMsa, seq::Character};
use linalg::{RowMatrix, Vector};

mod cpu;
mod cuda;
mod thread;

use cpu::CpuLikelihood;
use cuda::CudaLikelihood;
use thread::ThreadedLikelihood;

pub type Row<const N: usize> = Vector<f64, N>;
type Transition<const N: usize> = RowMatrix<f64, N, N>;

trait LikelihoodTrait<const N: usize> {
	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<N>],
		leaves_end: usize,
		root: usize,
	) -> Result<()>;

	fn likelihood(&mut self, weights: &[f64]) -> Result<f64>;

	fn accept(&mut self) -> Result<()>;

	fn reject(&mut self) -> Result<()>;
}

type DynCalculator<const N: usize> =
	Box<dyn LikelihoodTrait<N> + Send + Sync + 'static>;

pub struct GenericLikelihood<const N: usize> {
	substitution: BoxedSubstitutionModel<N>,
	clock: PyClock,
	transitions: Transitions<N>,
	calculator: DynCalculator<N>,
	weights: Vec<f64>,
	/// Last accepted likelihood
	cache: f64,
	/// Last calculated likelihood.  It's different from the cache, because
	/// it might get rejected.
	last: f64,
	launched_update: bool,
	tree: Py<PyTree>,

	unconstrained: f64,
}

impl GenericLikelihood<4> {
	fn new(
		substitution: BoxedSubstitutionModel<4>,
		clock: PyClock,
		msa: Msa<DnaNucleotide>,
		tree: Py<PyTree>,
		calculator: String,

		cuda_device: usize,
		thread_split_size: usize,
	) -> Result<Self> {
		let num_internals = msa.num_sequences() - 1;
		let transitions = Transitions::<4>::new(num_internals * 2);

		let (msa, weights) = deduplicate(msa);

		let calculator: DynCalculator<4> = match calculator.as_str() {
			"cpu" => Box::new(CpuLikelihood::new(msa)),
			"thread" => Box::new(ThreadedLikelihood::new(
				msa,
				thread_split_size,
			)),
			"cuda" => {
				Box::new(CudaLikelihood::new(msa, cuda_device)?)
			}
			_ => {
				bail!("Unknown calculator type '{calculator}'");
			}
		};

		let unconstrained = unconstrained(&weights);

		let mut out = Self {
			substitution,
			clock,
			transitions,
			calculator,
			weights,
			cache: f64::NAN,
			last: f64::NAN,
			launched_update: false,
			tree,
			unconstrained,
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

impl<const N: usize> GenericLikelihood<N> {
	fn propose(&mut self, py: Python) -> Result<()> {
		let tree = &mut self.tree.get().inner();
		let rate = self.clock.get_rate(py)?;
		let full_update = self.transitions.update(
			py,
			self.substitution.as_mut(),
			rate,
			tree,
		)?;
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

		self.calculator.propose(
			&nodes,
			&edges,
			&transitions,
			leaves_end,
			root,
		)?;
		self.launched_update = true;

		Ok(())
	}

	fn likelihood(&mut self) -> Result<f64> {
		if !self.launched_update {
			self.last = self.cache;
			return Ok(self.cache);
		}

		let likelihood = self.calculator.likelihood(&self.weights)?
			+ self.unconstrained;
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
		self.calculator.accept()?;
		self.transitions.accept();
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.calculator.reject()?;
		self.transitions.reject();
		Ok(())
	}
}

pub enum ErasedLikelihood {
	Nucleotide4(GenericLikelihood<4>),
	Nucleotide5(GenericLikelihood<5>),
	// TODO: amino: 20 standard, 2 special, stop codon
	Codon(GenericLikelihood<64>),
}

impl ErasedLikelihood {
	pub fn propose(&mut self, py: Python) -> Result<()> {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => {
				inner.propose(py)
			}
			_ => todo!(),
		}
	}

	pub fn likelihood(&mut self) -> Result<f64> {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => {
				inner.likelihood()
			}
			_ => todo!(),
		}
	}

	pub fn accept(&mut self) -> Result<()> {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => inner.accept(),
			ErasedLikelihood::Nucleotide5(inner) => inner.accept(),
			ErasedLikelihood::Codon(inner) => inner.accept(),
		}
	}

	pub fn reject(&mut self) -> Result<()> {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => inner.reject(),
			ErasedLikelihood::Nucleotide5(inner) => inner.reject(),
			ErasedLikelihood::Codon(inner) => inner.reject(),
		}
	}

	pub fn cached_likelihood(&self) -> f64 {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => inner.cache,
			ErasedLikelihood::Nucleotide5(inner) => inner.cache,
			ErasedLikelihood::Codon(inner) => inner.cache,
		}
	}
}

#[pyclass(name = "Likelihood", module = "aspartik.b3", frozen)]
pub struct PyLikelihood {
	inner: Mutex<ErasedLikelihood>,
}

impl PyLikelihood {
	pub fn inner(&self) -> MutexGuard<'_, ErasedLikelihood> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyLikelihood {
	#[new]
	#[pyo3(signature = (
		msa, substitution, clock, tree,
		calculator = String::from("cpu"),
		*,
		cuda_device = 0,
		thread_split_size = 400,
	))]
	fn new4(
		msa: PyMsa,
		substitution: BoxedSubstitutionModel<4>,
		clock: PyClock,
		tree: Py<PyTree>,
		calculator: String,

		cuda_device: usize,
		thread_split_size: usize,
	) -> Result<Self> {
		let generic_likelihood = GenericLikelihood::new(
			substitution,
			clock,
			msa.0,
			tree,
			calculator,
			cuda_device,
			thread_split_size,
		)?;

		let erased_likelihood =
			ErasedLikelihood::Nucleotide4(generic_likelihood);

		Ok(PyLikelihood {
			inner: Mutex::new(erased_likelihood),
		})
	}
}
