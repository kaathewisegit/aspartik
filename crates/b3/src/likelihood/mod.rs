use anyhow::Result;
use pyo3::prelude::*;

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
	state::PyState, substitution::PySubstitution, util::read_fasta,
	Transitions,
};
use linalg::{RowMatrix, Vector};

mod cpu;
mod gpu;
// mod thread;

use cpu::CpuLikelihood;
#[expect(unused)]
use gpu::GpuLikelihood;

// #[allow(unused)] // TODO: use dynamically in `State`
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

pub struct GenericLikelihood<const N: usize> {
	substitution: PySubstitution<N>,
	calculator: Box<dyn LikelihoodTrait<N> + Send + Sync>,
	cache: Option<f64>,
}

impl<const N: usize> GenericLikelihood<N> {
	pub fn propose(
		&mut self,
		py: Python,
		state: &PyState,
		transitions: &mut Transitions<N>,
	) -> Result<f64> {
		let substitution_matrix = self.substitution.get_matrix(py)?;
		let inner_state = state.inner();
		let tree = &*inner_state.tree.inner();
		let full_update = transitions.update(substitution_matrix, tree);
		let nodes = if full_update {
			tree.full_update()
		} else {
			tree.nodes_to_update()
		};

		// No update, we can return the last calculated value
		if nodes.is_empty() {
			return Ok(self.cache.unwrap());
		}

		let (nodes, edges, children) = tree.to_lists(&nodes);

		let transitions = transitions.matrices(&edges);

		Ok(self.calculator.propose(&nodes, &transitions, &children))
	}

	pub fn accept(&mut self) {
		self.calculator.accept();
	}

	pub fn reject(&mut self) {
		self.calculator.reject();
	}
}

pub enum Likelihood {
	Nucleotide4(GenericLikelihood<4>),
	Nucleotide5(GenericLikelihood<5>),
	// TODO: amino: 20 standard, 2 special, stop codon
	Codon(GenericLikelihood<64>),
}

impl Likelihood {
	pub fn propose(
		&mut self,
		py: Python,
		state: &PyState,
		// TODO: non-generic transitions wrapper
		transitions: &mut Transitions<4>,
	) -> Result<f64> {
		match self {
			Likelihood::Nucleotide4(inner) => {
				inner.propose(py, state, transitions)
			}
			_ => todo!(),
		}
	}

	pub fn accept(&mut self) {
		match self {
			Likelihood::Nucleotide4(inner) => inner.accept(),
			Likelihood::Nucleotide5(inner) => inner.accept(),
			Likelihood::Codon(inner) => inner.accept(),
		}
	}

	pub fn reject(&mut self) {
		match self {
			Likelihood::Nucleotide4(inner) => inner.accept(),
			Likelihood::Nucleotide5(inner) => inner.accept(),
			Likelihood::Codon(inner) => inner.accept(),
		}
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
	fn new4(data: &str, substitution: PySubstitution<4>) -> Result<Self> {
		let sites = read_fasta(data)?;
		let likelihood = Likelihood::Nucleotide4(GenericLikelihood {
			substitution,
			calculator: Box::new(CpuLikelihood::new(sites)),
			cache: None,
		});

		Ok(PyLikelihood {
			inner: Arc::new(Mutex::new(likelihood)),
		})
	}
}
