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

// TODO: should be generic over dimensions without any generics, somehow
pub struct Likelihood {
	substitution: PySubstitution<4>,
	calculator: Box<dyn LikelihoodTrait<4> + Send + Sync>,
}

impl Likelihood {
	pub fn propose(
		&mut self,
		py: Python,
		state: &PyState,
		transitions: &mut Transitions<4>,
	) -> Result<f64> {
		let substitution_matrix = self.substitution.get_matrix(py)?;
		let inner_state = state.inner();
		let tree = &*inner_state.tree.inner();
		let full_update = transitions.update(substitution_matrix, tree);
		let mut nodes = if full_update {
			tree.full_update()
		} else {
			tree.nodes_to_update()
		};

		// If there are no nodes to update, give us the root likelihood
		// XXX: can be optimized with caching
		if nodes.is_empty() {
			nodes.push(tree.root());
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
	fn new(data: &str, substitution: PySubstitution<4>) -> Result<Self> {
		let sites = read_fasta(data)?;
		let likelihood = Likelihood {
			substitution,
			calculator: Box::new(CpuLikelihood::new(sites)),
		};

		Ok(PyLikelihood {
			inner: Arc::new(Mutex::new(likelihood)),
		})
	}
}
