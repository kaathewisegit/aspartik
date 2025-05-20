#![allow(dead_code)]

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use pyo3::prelude::*;
use rand::Rng as _;

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

	fn run(&self, py: Python) -> Result<()> {
		for index in 0..self.length {
			self.step(py).with_context(|| {
				anyhow!("Failed on step {index}")
			})?;

			for logger in &self.loggers {
				logger.log(py, index).with_context(|| {
					anyhow!("Failed to log on step {index}")
				})?;
			}
		}

		Ok(())
	}
}

impl Mcmc {
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
					return Ok(());
				}
				Proposal::Hastings(ratio) => ratio,
			};

		for tree in &self.trees {
			tree.get().inner().verify()?;
		}

		let mut prior: f64 = 0.0;
		for py_prior in &self.priors {
			prior += py_prior.probability(py)?;

			// short-circuit on a rejection by any prior
			if prior == f64::NEG_INFINITY {
				self.reject()?;
				return Ok(());
			}
		}

		let mut likelihood = 0.0;
		for py_likelihood in &self.likelihoods {
			likelihood +=
				py_likelihood.get().inner().propose(py)?;
		}
		let new_posterior = likelihood + prior;

		let old_posterior = *self.posterior.lock();

		let ratio = new_posterior - old_posterior + hastings;

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
