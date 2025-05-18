use anyhow::Result;
use pyo3::prelude::*;

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
	substitution::PySubstitution,
	tree::{PyTree, Tree},
	util::{dna_to_rows, read_fasta},
	Transitions,
};
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
	cache: Option<f64>,
}

impl GenericLikelihood<4> {
	fn new(
		substitution: PySubstitution<4>,
		sites: Vec<Vec<Vector<f64, 4>>>,
	) -> Self {
		let num_internals = sites[0].len() - 1;
		let transitions = Transitions::<4>::new(num_internals * 2);

		let size = sites[0].len() * sites.len();
		// XXX: establish a heuristic
		let calculator: DynCalculator<4> = if size > 100_000 {
			Box::new(GpuLikelihood::new(sites))
		} else {
			Box::new(CpuLikelihood::new(sites))
		};

		Self {
			substitution,
			transitions,
			calculator,
			cache: None,
		}
	}
}

impl<const N: usize> GenericLikelihood<N> {
	fn propose(&mut self, py: Python, tree: &Tree) -> Result<f64> {
		let substitution_matrix = self.substitution.get_matrix(py)?;
		let full_update =
			self.transitions.update(substitution_matrix, tree);
		let nodes = if full_update || self.cache.is_none() {
			tree.full_update()
		} else {
			tree.nodes_to_update()
		};

		// No update, we can return the last calculated value
		if nodes.is_empty() {
			// we can unwrap here because on the first calculation
			// (no likelihood yet) we'll do a full update
			return Ok(self.cache.unwrap());
		}

		let (nodes, edges, children) = tree.to_lists(&nodes);

		let transitions = self.transitions.matrices(&edges);

		Ok(self.calculator.propose(&nodes, &transitions, &children))
	}

	fn accept(&mut self) {
		self.calculator.accept();
	}

	fn reject(&mut self) {
		self.calculator.reject();
	}
}

pub enum ErasedLikelihood {
	Nucleotide4(GenericLikelihood<4>),
	Nucleotide5(GenericLikelihood<5>),
	// TODO: amino: 20 standard, 2 special, stop codon
	Codon(GenericLikelihood<64>),
}

impl ErasedLikelihood {
	pub fn propose(&mut self, py: Python, tree: &Tree) -> Result<f64> {
		match self {
			ErasedLikelihood::Nucleotide4(inner) => {
				inner.propose(py, tree)
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
}

pub struct Likelihood {
	erased: ErasedLikelihood,
	tree: PyTree,
}

impl Likelihood {
	pub fn propose(&mut self, py: Python) -> Result<f64> {
		self.erased.propose(py, &self.tree.inner())
	}

	pub fn accept(&mut self) {
		self.erased.accept();
	}

	pub fn reject(&mut self) {
		self.erased.reject();
	}
}

#[derive(Clone)]
#[pyclass(name = "Likelihood", module = "aspartik.b3", frozen)]
pub struct PyLikelihood {
	inner: Arc<Mutex<Likelihood>>,
}

impl PyLikelihood {
	pub fn inner(&self) -> MutexGuard<Likelihood> {
		self.inner.lock().unwrap()
	}
}

#[pymethods]
impl PyLikelihood {
	#[new]
	fn new4(
		data: &str,
		substitution: PySubstitution<4>,
		tree: PyTree,
	) -> Result<Self> {
		let seqs = read_fasta(data)?;
		let sites = dna_to_rows(&seqs);

		let erased_likelihood = ErasedLikelihood::Nucleotide4(
			GenericLikelihood::new(substitution, sites),
		);

		let likelihood = Likelihood {
			erased: erased_likelihood,
			tree,
		};

		Ok(PyLikelihood {
			inner: Arc::new(Mutex::new(likelihood)),
		})
	}
}
