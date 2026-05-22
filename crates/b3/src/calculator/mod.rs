use anyhow::Result;
use parking_lot::MutexGuard;
use pyo3::{prelude::*, types::PyType};

use std::borrow::BorrowMut;

use crate::{Transitions, parameters::Tree};

mod cpu4;
mod cuda;

pub use cpu4::Cpu4Calculator;
pub use cuda::CudaCalculator;

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

#[derive(Debug, Clone, Copy)]
enum CalculatorKind {
	Cpu { num_threads: usize },
	Cuda { device: usize },
}

#[derive(Debug, Clone, Copy)]
#[pyclass(name = "Calculator", module = "aspartik.b3", from_py_object)]
pub struct CalculatorConfig {
	kind: CalculatorKind,
	scale_ln: u32,
}

#[pymethods]
impl CalculatorConfig {
	#[pyo3(name = "CPU", signature = (num_threads = 0))]
	#[classmethod]
	fn cpu(_cls: Py<PyType>, num_threads: usize) -> Self {
		Self {
			kind: CalculatorKind::Cpu { num_threads },
			scale_ln: 30,
		}
	}

	#[pyo3(name = "CUDA", signature = (device = 0))]
	#[classmethod]
	fn cuda(_cls: Py<PyType>, device: usize) -> Self {
		Self {
			kind: CalculatorKind::Cuda { device },
			scale_ln: 30,
		}
	}

	fn with_scale(
		mut this: PyRefMut<'_, Self>,
		scale_ln: u32,
	) -> PyRefMut<'_, Self> {
		this.borrow_mut().scale_ln = scale_ln;
		this
	}
}

impl CalculatorConfig {
	pub fn make4(
		&self,
		samples: Vec<u8>,
		weights: Vec<u32>,
	) -> Result<Box<dyn Calculator<4, f64> + Send>> {
		match self.kind {
			CalculatorKind::Cpu { num_threads } => {
				let calc = Cpu4Calculator::new(
					weights,
					samples,
					self.scale_ln,
					num_threads,
				);
				Ok(Box::new(calc))
			}
			CalculatorKind::Cuda { device } => {
				let calc = CudaCalculator::new(
					weights,
					samples,
					self.scale_ln,
					device,
				)?;
				Ok(Box::new(calc))
			}
		}
	}
}
