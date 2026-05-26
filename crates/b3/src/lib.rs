pub mod calculator;
pub mod callback;
pub mod clock;
pub mod likelihood;
pub mod mcmc;
pub mod operators;
pub mod parameters;
pub mod priors;
mod sk_slice_buf;
pub mod substitution;
mod transitions;

pub use callback::PyCallback;
pub use sk_slice_buf::SkSliceBuf;
pub use transitions::Transitions;

use pyo3::prelude::*;

#[pymodule(name = "_b3_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use crate::{
		calculator::CalculatorConfig,
		callback::TraceWriter,
		clock::PyClock,
		likelihood::{
			CompoundLikelihood, DnaLikelihood, GammaLikelihood,
			HeteroLikelihood,
		},
		mcmc::Mcmc,
		operators::{
			ClassvecFlip, FixedHeightSPR, Proposal, SubtreeLeap,
		},
		parameters::{
			Internal, Leaf, PyClassVector, PyIntVector, PyReal,
			PyRealVector, PyTree,
		},
		priors::{
			BayesianSkyline, ConstantPopulation, ExponentialGrowth,
			Monophyly, SymmetricDirichlet, Yule,
		},
		substitution::{GTR, HKY, JC, K80},
	};

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		util::py_patch_module!(m);

		Ok(())
	}
}
