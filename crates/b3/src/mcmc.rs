use anyhow::{anyhow, Context, Result};
use log::trace;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng as _;

use crate::{
	likelihood::PyLikelihood,
	operator::{Proposal, PyOperator, WeightedScheduler},
	PyLogger, PyPrior,
};
use rng::PyRng;
use util::py_call_method;

#[pyclass(name = "MCMC", module = "aspartik.b3", frozen)]
pub struct Mcmc {
	posterior: Mutex<f64>,

	current_step: Mutex<usize>,
	burnin: usize,
	length: usize,

	state: Vec<PyObject>,
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
	#[pyo3(signature = (
		burnin, length,
		state, priors, operators, likelihoods, loggers, rng,
	))]
	fn new(
		py: Python,

		burnin: usize,
		length: usize,

		state: Vec<PyObject>,
		priors: Vec<PyPrior>,
		operators: Vec<PyOperator>,
		likelihoods: Vec<Py<PyLikelihood>>,
		loggers: Vec<PyLogger>,
		rng: Py<PyRng>,
	) -> Result<Mcmc> {
		let scheduler = WeightedScheduler::new(py, operators)?;

		Ok(Mcmc {
			posterior: Mutex::new(f64::NEG_INFINITY),
			current_step: Mutex::new(0),

			burnin,
			length,

			state,
			priors,
			scheduler,
			likelihoods,
			loggers,
			rng,
		})
	}

	#[getter]
	fn current_step(&self) -> usize {
		*self.current_step.lock()
	}

	#[getter]
	fn state(&self, py: Python) -> Vec<PyObject> {
		self.state.iter().map(|s| s.clone_ref(py)).collect()
	}

	#[getter]
	fn priors(&self, py: Python) -> Vec<PyObject> {
		self.priors.iter().map(|p| p.clone_ref(py)).collect()
	}

	#[getter]
	fn likelihoods(&self, py: Python) -> Vec<Py<PyLikelihood>> {
		self.likelihoods.iter().map(|l| l.clone_ref(py)).collect()
	}

	#[getter]
	fn loggers(&self, py: Python) -> Vec<PyObject> {
		self.loggers.iter().map(|l| l.clone_ref(py)).collect()
	}

	#[getter]
	fn rng(&self, py: Python) -> Py<PyRng> {
		self.rng.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<PyObject> {
		let tuple = (
			self.burnin,
			self.length,
			self.state(py),
			self.priors(py),
			// TODO: operators
			PyList::empty(py),
			self.likelihoods(py),
			self.loggers(py),
			self.rng(py),
		)
			.into_pyobject(py)?;

		Ok(tuple.into_any().unbind())
	}

	fn run(this: Py<Self>, py: Python) -> Result<()> {
		let self_ = this.get();
		loop {
			let current_step = *self_.current_step.lock();
			if current_step == self_.length {
				break;
			}

			self_.step(py).with_context(|| {
				anyhow!("Failed on step {current_step}")
			})?;

			if current_step >= self_.burnin {
				Self::log(
					this.clone_ref(py),
					py,
					current_step,
				)?;
			}

			*self_.current_step.lock() += 1;
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
	fn step(&self, py: Python) -> Result<()> {
		let rng = self.rng.get();
		let operator = self.scheduler.select_operator(&mut rng.inner());

		let proposal = operator.propose(py).with_context(|| {
			anyhow!(
				"Operator {} failed while generating a proposal",
				operator.repr(py).unwrap()
			)
		})?;
		let hastings = match proposal {
			Proposal::Accept() => {
				self.accept(py)?;
				return Ok(());
			}
			Proposal::Reject() => {
				self.reject(py)?;
				return Ok(());
			}
			Proposal::Hastings(ratio) => ratio,
		};

		let prior = self.prior(py)?;
		// The proposal will be rejected regardless of likelihood
		if prior == f64::NEG_INFINITY {
			self.reject(py)?;
			return Ok(());
		}

		// Update likelihoods.
		for py_likelihood in &self.likelihoods {
			py_likelihood.get().inner().propose(py)?;
		}

		// Collect the resulting likelihoods.  This is done separately
		// from proposing to allow launching parallel workloads.  A user
		// might, for example, use two CUDA devices.  Then `propose`
		// will queue both of them asynchronously and `likelihood` will
		// wait for completion of both.
		let mut likelihood = 0.0;
		for py_likelihood in &self.likelihoods {
			likelihood +=
				py_likelihood.get().inner().likelihood()?;
		}

		let new_posterior = likelihood + prior;

		let old_posterior = *self.posterior.lock();

		let ratio = new_posterior - old_posterior + hastings;

		trace!(
			target: "b3::mcmc::step",
			likelihood, prior, hastings,
			new_posterior, old_posterior, ratio;
			""
		);

		let random_0_1 = self.rng.get().inner().random::<f64>();
		if ratio > random_0_1.ln() {
			*self.posterior.lock() = new_posterior;

			self.accept(py)?;
		} else {
			self.reject(py)?;
		}

		Ok(())
	}

	fn accept(&self, py: Python) -> Result<()> {
		trace!(target: "b3::mcmc", "accept");

		self.scheduler.accept();

		for likelihood in &self.likelihoods {
			likelihood.get().inner().accept()?;
		}

		for parameter in &self.state {
			py_call_method!(py, parameter, "accept")?;
		}

		Ok(())
	}

	fn reject(&self, py: Python) -> Result<()> {
		trace!(target: "b3::mcmc", "reject");

		self.scheduler.reject();

		for likelihood in &self.likelihoods {
			likelihood.get().inner().reject()?;
		}

		for parameter in &self.state {
			py_call_method!(py, parameter, "reject")?;
		}

		Ok(())
	}

	fn log(this: Py<Self>, py: Python, current_step: usize) -> Result<()> {
		let self_ = this.get();

		for logger in &self_.loggers {
			if !logger.should_log(current_step) {
				continue;
			}

			let result = logger.log(py, this.clone_ref(py));
			result.with_context(|| {
				anyhow!("Failed to log on step {current_step}")
			})?;
		}

		Ok(())
	}
}
