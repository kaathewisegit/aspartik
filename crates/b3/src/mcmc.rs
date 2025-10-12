use anyhow::{Context, Result, anyhow};
use log::trace;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng as _;

use std::time::Instant;

use crate::{
	PyCallback, PyPrior,
	likelihood::PyLikelihood,
	operator::{Proposal, PyOperator, WeightedScheduler},
};
use rng::PyRng;
use util::py_call_method;

#[pyclass(name = "MCMC", module = "aspartik.b3", frozen)]
pub struct Mcmc {
	posterior: Mutex<f64>,

	current_step: Mutex<usize>,
	burnin: usize,
	length: usize,

	state: Vec<Py<PyAny>>,
	priors: Vec<PyPrior>,
	scheduler: WeightedScheduler,
	likelihoods: Vec<Py<PyLikelihood>>,
	callbacks: Vec<PyCallback>,
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
		state, priors, operators, likelihoods, callbacks, rng,
	))]
	fn new(
		py: Python,

		burnin: usize,
		length: usize,

		state: Vec<Py<PyAny>>,
		priors: Vec<PyPrior>,
		operators: Vec<PyOperator>,
		likelihoods: Vec<Py<PyLikelihood>>,
		callbacks: Vec<PyCallback>,
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
			callbacks,
			rng,
		})
	}

	#[getter]
	fn current_step(&self) -> usize {
		*self.current_step.lock()
	}

	#[getter]
	fn state(&self, py: Python) -> Vec<Py<PyAny>> {
		self.state.iter().map(|s| s.clone_ref(py)).collect()
	}

	#[getter]
	fn priors(&self, py: Python) -> Vec<Py<PyAny>> {
		self.priors.iter().map(|p| p.clone_ref(py)).collect()
	}

	#[getter]
	fn operators(&self, py: Python) -> Vec<Py<PyAny>> {
		self.scheduler.operators(py)
	}

	#[getter]
	fn likelihoods(&self, py: Python) -> Vec<Py<PyLikelihood>> {
		self.likelihoods.iter().map(|l| l.clone_ref(py)).collect()
	}

	#[getter]
	fn callbacks(&self, py: Python) -> Vec<Py<PyAny>> {
		self.callbacks.iter().map(|l| l.clone_ref(py)).collect()
	}

	#[getter]
	fn rng(&self, py: Python) -> Py<PyRng> {
		self.rng.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyAny>> {
		let tuple = (
			self.burnin,
			self.length,
			self.state(py),
			self.priors(py),
			self.operators(py),
			self.likelihoods(py),
			self.callbacks(py),
			self.rng(py),
		)
			.into_pyobject(py)?;

		Ok(tuple.into_any().unbind())
	}

	fn run(this: Py<Self>, py: Python) -> Result<()> {
		let self_ = this.get();
		loop {
			let current_step = *self_.current_step.lock();

			self_.step(py).with_context(|| {
				anyhow!("Failed on step {current_step}")
			})?;

			if current_step >= self_.burnin {
				Self::call_callbacks(
					this.clone_ref(py),
					py,
					current_step,
				)?;
			}

			if current_step == self_.length {
				break;
			}
			*self_.current_step.lock() += 1;
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

	#[getter]
	fn operator_statistics(&self, py: Python) -> Result<Py<PyList>> {
		self.scheduler.statistics(py)
	}
}

impl Mcmc {
	fn step(&self, py: Python) -> Result<()> {
		let rng = self.rng.get();
		let operator_index =
			self.scheduler.random_operator_index(&mut rng.inner());
		let operator = self.scheduler.get_operator(operator_index);

		let propose_start = Instant::now();
		let proposal = operator.propose(py).with_context(|| {
			anyhow!(
				"Operator {} failed while generating a proposal",
				operator.repr(py).unwrap()
			)
		})?;
		self.scheduler.record_propose_duration(
			operator_index,
			Instant::now() - propose_start,
		);

		let hastings = match proposal {
			Proposal::Accept() => {
				self.accept(py, operator_index)?;
				return Ok(());
			}
			Proposal::Reject() => {
				self.reject(py, operator_index)?;
				return Ok(());
			}
			Proposal::Hastings(ratio) => ratio,
		};

		let prior = self.prior(py)?;
		// The proposal will be rejected regardless of likelihood
		if prior == f64::NEG_INFINITY {
			self.reject(py, operator_index)?;
			return Ok(());
		}

		let likelihood_start = Instant::now();

		// Update likelihoods.
		for py_likelihood in &self.likelihoods {
			py_likelihood.get().inner().propose(py)?;
		}

		self.scheduler.record_likelihood_duration(
			operator_index,
			Instant::now() - likelihood_start,
		);

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

			self.accept(py, operator_index)?;
		} else {
			self.reject(py, operator_index)?;
		}

		Ok(())
	}

	fn accept(&self, py: Python, operator_index: usize) -> Result<()> {
		trace!(target: "b3::mcmc", "accept");

		self.scheduler.accept(py, operator_index)?;

		for likelihood in &self.likelihoods {
			likelihood.get().inner().accept()?;
		}

		for parameter in &self.state {
			py_call_method!(py, parameter, "accept")?;
		}

		Ok(())
	}

	fn reject(&self, py: Python, operator_index: usize) -> Result<()> {
		trace!(target: "b3::mcmc", "reject");

		self.scheduler.reject(py, operator_index)?;

		for likelihood in &self.likelihoods {
			likelihood.get().inner().reject()?;
		}

		for parameter in &self.state {
			py_call_method!(py, parameter, "reject")?;
		}

		Ok(())
	}

	fn call_callbacks(
		this: Py<Self>,
		py: Python,
		current_step: usize,
	) -> Result<()> {
		let self_ = this.get();

		for callback in &self_.callbacks {
			if !callback.should_call(current_step) {
				continue;
			}

			let result = callback.call(py, this.clone_ref(py));
			result.with_context(|| {
				anyhow!("Failed to log on step {current_step}")
			})?;
		}

		Ok(())
	}
}
