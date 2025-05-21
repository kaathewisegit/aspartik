use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use pyo3::prelude::*;
use rand::Rng as _;
use tracing::{instrument, trace};

use crate::{
	likelihood::PyLikelihood,
	operator::{Proposal, PyOperator, WeightedScheduler},
	parameter::{Parameter, PyParameter},
	tree::PyTree,
	PyLogger, PyPrior,
};
use rng::PyRng;

#[pyclass(name = "MCMC", module = "aspartik.b3", frozen)]
pub struct Mcmc {
	posterior: Mutex<f64>,

	#[allow(unused)]
	burnin: usize,
	length: usize,

	trees: Vec<Py<PyTree>>,

	/// TODO: parameter serialization
	backup_params: Mutex<Vec<Parameter>>,
	/// Current set of parameters by name.
	params: Vec<PyParameter>,

	priors: Vec<PyPrior>,
	scheduler: WeightedScheduler,
	likelihoods: Vec<Py<PyLikelihood>>,
	loggers: Vec<PyLogger>,
	rng: Py<PyRng>,
}

#[pymethods]
impl Mcmc {
	// This is a big constructor, so all of the arguments have to be here.
	// In theory it might make sense to join trees and parameters together,
	// but I'll have to benchmark that.
	#[expect(clippy::too_many_arguments)]
	#[new]
	fn new(
		py: Python,

		burnin: usize,
		length: usize,

		trees: Vec<Py<PyTree>>,
		params: Vec<PyParameter>,
		priors: Vec<PyPrior>,
		operators: Vec<PyOperator>,
		likelihoods: Vec<Py<PyLikelihood>>,
		loggers: Vec<PyLogger>,
		rng: Py<PyRng>,
	) -> Result<Mcmc> {
		let mut backup_params = Vec::with_capacity(params.len());
		for param in &params {
			backup_params.push(param.inner().clone());
		}
		let backup_params = Mutex::new(backup_params);
		let scheduler = WeightedScheduler::new(py, operators)?;

		Ok(Mcmc {
			posterior: Mutex::new(f64::NEG_INFINITY),
			burnin,
			length,
			trees,
			params,
			backup_params,
			priors,
			scheduler,
			likelihoods,
			loggers,
			rng,
		})
	}

	#[instrument(skip_all)]
	fn run(this: Py<Self>, py: Python) -> Result<()> {
		let self_ = this.get();
		for index in 0..self_.length {
			trace!(index);
			self_.step(py).with_context(|| {
				anyhow!("Failed on step {index}")
			})?;

			for logger in &self_.loggers {
				logger.log(py, this.clone_ref(py), index)
					.with_context(|| {
						anyhow!("Failed to log on step {index}")
					})?;
			}
		}

		Ok(())
	}

	#[getter]
	fn posterior(&self) -> f64 {
		*self.posterior.lock()
	}

	#[getter]
	fn likelihood(&self) -> f64 {
		let mut out = 0.0;
		for likelihood in &self.likelihoods {
			out += likelihood.get().inner().cached_likelihood();
		}
		out
	}

	#[getter]
	fn prior(&self, py: Python) -> Result<f64> {
		let mut out = 0.0;
		for py_prior in &self.priors {
			out += py_prior.probability(py)?;

			// short-circuit on a rejection by any prior
			if out == f64::NEG_INFINITY {
				return Ok(out);
			}
		}
		Ok(out)
	}
}

impl Mcmc {
	#[instrument(skip_all)]
	fn step(&self, py: Python) -> Result<()> {
		let rng = self.rng.get();
		let operator = self.scheduler.select_operator(&mut rng.inner());

		let hastings =
			match operator.propose(py).with_context(|| {
				anyhow!(
			"Operator {} failed while generating a proposal",
			operator.repr(py).unwrap()
		)
			})? {
				Proposal::Accept() => {
					trace!("accept proposal");
					self.accept()?;
					return Ok(());
				}
				Proposal::Reject() => {
					trace!("reject proposal");
					self.reject()?;
					return Ok(());
				}
				Proposal::Hastings(ratio) => ratio,
			};

		for tree in &self.trees {
			tree.get().inner().verify()?;
		}

		let prior = self.prior(py)?;
		// The proposal will be rejected regardless of likelihood
		if prior == f64::NEG_INFINITY {
			trace!("reject proposal");
			self.reject()?;
			return Ok(());
		}

		let mut likelihood = 0.0;
		for py_likelihood in &self.likelihoods {
			likelihood +=
				py_likelihood.get().inner().propose(py)?;
		}
		let new_posterior = likelihood + prior;

		let old_posterior = *self.posterior.lock();

		let ratio = new_posterior - old_posterior + hastings;

		trace!(
			likelihood,
			prior,
			new_posterior,
			old_posterior,
			hastings,
			ratio
		);

		let random_0_1 = self.rng.get().inner().random::<f64>();
		if ratio > random_0_1.ln() {
			*self.posterior.lock() = new_posterior;

			trace!("accept proposal");
			self.accept()?;
		} else {
			trace!("reject proposal");
			self.reject()?;
		}

		Ok(())
	}

	fn accept(&self) -> Result<()> {
		for tree in &self.trees {
			tree.get().inner().accept();
		}

		for likelihood in &self.likelihoods {
			likelihood.get().inner().accept();
		}

		let mut backup_params = self.backup_params.lock();
		for i in 0..self.params.len() {
			backup_params[i] = self.params[i].inner().clone();
		}

		Ok(())
	}

	fn reject(&self) -> Result<()> {
		for tree in &self.trees {
			tree.get().inner().reject();
		}

		for likelihood in &self.likelihoods {
			likelihood.get().inner().reject();
		}

		let backup_params = self.backup_params.lock();
		for i in 0..self.params.len() {
			*self.params[i].inner() = backup_params[i].clone();
		}

		Ok(())
	}
}
