use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use pyo3::prelude::*;
use rand::Rng as _;
use tracing::{instrument, trace};

use crate::{
	likelihood::PyLikelihood,
	operator::{Proposal, PyOperator, WeightedScheduler},
	parameter::PyParameter,
	tree::PyTree,
	PyLogger, PyPrior,
};
use rng::PyRng;

#[pyclass(name = "MCMC", module = "aspartik.b3", frozen)]
pub struct Mcmc {
	posterior: Mutex<f64>,

	#[expect(unused)]
	burnin: usize,
	length: usize,

	trees: Vec<Py<PyTree>>,

	/// Current set of parameters by name.
	params: Vec<Py<PyParameter>>,

	priors: Vec<PyPrior>,
	scheduler: WeightedScheduler,
	likelihoods: Vec<Py<PyLikelihood>>,
	loggers: Vec<PyLogger>,
	rng: Py<PyRng>,

	// Config
	validate: bool,
}

#[pymethods]
impl Mcmc {
	// This is a big constructor, so all of the arguments have to be here.
	// In theory it might make sense to join trees and parameters together,
	// but I'll have to benchmark that.
	#[expect(clippy::too_many_arguments)]
	#[new]
	#[pyo3(signature = (
		burnin, length,
		trees, params, priors, operators, likelihoods, loggers, rng,
		validate = false,
	))]
	fn new(
		py: Python,

		burnin: usize,
		length: usize,

		trees: Vec<Py<PyTree>>,
		params: Vec<Py<PyParameter>>,
		priors: Vec<PyPrior>,
		operators: Vec<PyOperator>,
		likelihoods: Vec<Py<PyLikelihood>>,
		loggers: Vec<PyLogger>,
		rng: Py<PyRng>,

		validate: bool,
	) -> Result<Mcmc> {
		let scheduler = WeightedScheduler::new(py, operators)?;

		Ok(Mcmc {
			posterior: Mutex::new(f64::NEG_INFINITY),

			burnin,
			length,

			trees,
			params,
			priors,
			scheduler,
			likelihoods,
			loggers,
			rng,

			validate,
		})
	}

	#[instrument(skip_all)]
	fn run(this: Py<Self>, py: Python) -> Result<()> {
		let self_ = this.get();
		for index in 0..self_.length {
			trace!(step = index);
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

		self_.scheduler.report(py)?;

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
					self.accept()?;
					return Ok(());
				}
				Proposal::Reject() => {
					self.reject()?;
					return Ok(());
				}
				Proposal::Hastings(ratio) => ratio,
			};

		if self.validate {
			for tree in &self.trees {
				tree.get().inner().validate()?;
			}
		}

		let prior = self.prior(py)?;
		// The proposal will be rejected regardless of likelihood
		if prior == f64::NEG_INFINITY {
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

			self.accept()?;
		} else {
			self.reject()?;
		}

		Ok(())
	}

	fn accept(&self) -> Result<()> {
		trace!("accept proposal");

		self.scheduler.accept();

		for tree in &self.trees {
			tree.get().inner().accept();
		}

		for likelihood in &self.likelihoods {
			likelihood.get().inner().accept()?;
		}

		for i in 0..self.params.len() {
			self.params[i].get().accept();
		}

		Ok(())
	}

	fn reject(&self) -> Result<()> {
		trace!("reject proposal");

		self.scheduler.reject();

		for tree in &self.trees {
			tree.get().inner().reject();
		}

		for likelihood in &self.likelihoods {
			likelihood.get().inner().reject()?;
		}

		for i in 0..self.params.len() {
			self.params[i].get().reject();
		}

		Ok(())
	}
}
