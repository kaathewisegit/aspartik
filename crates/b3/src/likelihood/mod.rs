use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use tracing::{info, instrument, trace};

use crate::{
	substitution::PySubstitution, tree::PyTree, util::dna_to_rows,
	Transitions,
};
use data::seq::{python::PyDnaSeq, DnaSeq};
use linalg::{RowMatrix, Vector};

mod cpu;
mod gpu;
// mod thread;

use cpu::CpuLikelihood;
use gpu::GpuLikelihood;

// pub use thread::ThreadedLikelihood;

pub type Row<const N: usize> = Vector<f64, N>;
type Transition<const N: usize> = RowMatrix<f64, N, N>;

trait LikelihoodTrait<const N: usize> {
	fn propose(
		&mut self,
		nodes: &[usize],
		transitions: &[Transition<N>],
		children: &[usize],
	) -> f64;

	fn accept(&mut self);

	fn reject(&mut self);
}

type DynCalculator<const N: usize> =
	Box<dyn LikelihoodTrait<N> + Send + Sync + 'static>;

pub struct GenericLikelihood<const N: usize> {
	substitution: PySubstitution<N>,
	transitions: Transitions<N>,
	calculator: DynCalculator<N>,
	/// Last accepted likelihood
	cache: f64,
	/// Last calculated likelihood.  It's different from the cache, because
	/// it might get rejected.
	last: f64,
	tree: Py<PyTree>,
}

impl GenericLikelihood<4> {
	#[instrument(skip_all, fields(use_gpu))]
	fn new(
		substitution: PySubstitution<4>,
		sites: Vec<Vec<Vector<f64, 4>>>,
		tree: Py<PyTree>,
		use_gpu: bool,
	) -> Result<Self> {
		let num_internals = sites[0].len() - 1;
		let transitions = Transitions::<4>::new(num_internals * 2);

		let calculator: DynCalculator<4> = if use_gpu {
			info!("using GpuLikelihood");
			Box::new(GpuLikelihood::new(sites)?)
		} else {
			info!("using CpuLikelihood");
			Box::new(CpuLikelihood::new(sites))
		};

		let mut out = Self {
			substitution,
			transitions,
			calculator,
			tree,
			cache: f64::NAN,
			last: f64::NAN,
		};
		Python::with_gil(|py| out.propose(py))?;
		// propose sets `last` and accept updates the cache, so neither
		// cache nor last will be NaN.
		out.accept();
		Ok(out)
	}
}

impl<const N: usize> GenericLikelihood<N> {
	#[instrument(skip_all)]
	fn propose(&mut self, py: Python) -> Result<f64> {
		let tree = &self.tree.get().inner();
		let substitution_matrix = self.substitution.get_matrix(py)?;
		let full_update =
			self.transitions.update(substitution_matrix, tree);
		let nodes = if full_update {
			tree.full_update()
		} else {
			tree.nodes_to_update()
		};
		trace!(num_nodes_to_update = nodes.len());

		// no tree update, return the cache
		if nodes.is_empty() {
			self.last = self.cache;
			return Ok(self.cache);
		}

		let (nodes, edges, children) = tree.to_lists(&nodes);

		let transitions = self.transitions.matrices(&edges);

		let likelihood = self.calculator.propose(
			&nodes,
			&transitions,
			&children,
		);
		trace!(likelihood);
		self.last = likelihood;
		Ok(likelihood)
	}

	fn accept(&mut self) {
		self.cache = self.last;
		self.calculator.accept();
		self.transitions.accept();
	}

	fn reject(&mut self) {
		self.calculator.reject();
		self.transitions.accept();
	}
}

pub enum ErasedLikelihood {
	Nucleotide4(GenericLikelihood<4>),
	Nucleotide5(GenericLikelihood<5>),
	// TODO: amino: 20 standard, 2 special, stop codon
	Codon(GenericLikelihood<64>),
}

impl ErasedLikelihood {
	pub fn propose(&mut self, py: Python) -> Result<f64> {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => {
				inner.propose(py)
			}
			_ => todo!(),
		}
	}

	pub fn accept(&mut self) {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => inner.accept(),
			ErasedLikelihood::Nucleotide5(inner) => inner.accept(),
			ErasedLikelihood::Codon(inner) => inner.accept(),
		}
	}

	pub fn reject(&mut self) {
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
	pub fn inner(&self) -> MutexGuard<ErasedLikelihood> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyLikelihood {
	#[new]
	#[pyo3(signature = (sequences, substitution, tree, use_gpu = false))]
	fn new4(
		sequences: Vec<PyDnaSeq>,
		substitution: PySubstitution<4>,
		tree: Py<PyTree>,
		use_gpu: bool,
	) -> Result<Self> {
		let sequences: Vec<DnaSeq> = sequences
			.iter()
			.map(|seq| seq.clone().into())
			.collect();
		let sites = dna_to_rows(&sequences);

		let generic_likelihood = GenericLikelihood::new(
			substitution,
			sites,
			tree,
			use_gpu,
		)?;

		let erased_likelihood =
			ErasedLikelihood::Nucleotide4(generic_likelihood);

		Ok(PyLikelihood {
			inner: Mutex::new(erased_likelihood),
		})
	}
}
