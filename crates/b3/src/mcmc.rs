use anyhow::{Context, Result, anyhow};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng as _;

use crate::{
	PyCallback, PyPrior,
	likelihood::PyLikelihood,
	operator::{Proposal, PyOperator, WeightedScheduler},
};
use profiler::time;
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

			let operator_index =
				self_.scheduler.random_operator_index(
					&mut self_.rng.get().inner(),
				);

			let result = self_
				.step(py, operator_index)
				.with_context(|| {
					anyhow!("Failed on step {current_step}")
				})?;

			self_.finalize(py, operator_index, result)?;

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

	fn measure_operator(
		&self,
		py: Python,
		operator_index: usize,
		length: usize,
	) -> Result<[usize; 5]> {
		let mut out = [0; 5];

		for _ in 0..length {
			let result = self.step(py, operator_index)?;
			self.finalize(py, operator_index, StepResult::Reject)?;
			out[result.index()] += 1;
		}

		Ok(out)
	}

	#[getter]
	fn posterior(&self) -> f64 {
		*self.posterior.lock()
	}

	#[getter]
	fn cached_likelihood(&self) -> f64 {
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

#[derive(Debug, Clone, Copy)]
pub enum StepResult {
	/// Operator returned `Proposal::Reject`
	UnconditionalAccept = 0,
	/// Operator returned `Proposal::Accept`
	UnconditionalReject,
	/// A prior returned negative infinity
	PriorReject,
	/// Regular MCMC accept
	Accept,
	/// Regular MCMC reject
	Reject,
}

impl StepResult {
	pub fn is_accept(&self) -> bool {
		matches!(self, Self::UnconditionalAccept | Self::Accept)
	}

	pub fn is_reject(&self) -> bool {
		!self.is_accept()
	}

	pub fn index(&self) -> usize {
		*self as usize
	}
}

impl Mcmc {
	/// Triggers likelihood recalculation
	///
	/// The calculations might be asynchronous and done in parallel.
	/// Calling `calculate_likelihood` will await the results.
	fn propose(&self, py: Python) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.get().inner().propose(py)?;
		}
		Ok(())
	}

	/// Await the calculations of all likelihoods
	///
	/// This must be called after the `propose` method.  Calling it without
	/// calling `propose` or calling it will lead to a deadlock (`thread`
	/// calculator) or an incorrect value.
	fn calculate_likelihood(&self) -> Result<f64> {
		self.likelihoods
			.iter()
			.map(|likelihood| likelihood.get().inner().likelihood())
			.sum()
	}

	fn step(
		&self,
		py: Python,
		operator_index: usize,
	) -> Result<StepResult> {
		use StepResult::*;

		let proposal =
			self.scheduler.make_proposal(py, operator_index)?;

		let hastings = match proposal {
			Proposal::Accept() => {
				return Ok(UnconditionalAccept);
			}
			Proposal::Reject() => {
				return Ok(UnconditionalReject);
			}
			Proposal::Hastings(ratio) => ratio,
		};

		let prior = self.prior(py)?;
		// The proposal will be rejected regardless of likelihood
		if prior == f64::NEG_INFINITY {
			return Ok(PriorReject);
		}

		let (likelihood, time) = time! {{
			self.propose(py)?;
			self.calculate_likelihood()?
		}};

		self.scheduler
			.record_likelihood_duration(operator_index, time);

		let new_posterior = likelihood + prior;

		let old_posterior = *self.posterior.lock();

		let ratio = new_posterior - old_posterior + hastings;

		let random_0_1 = self.rng.get().inner().random::<f64>();
		if ratio > random_0_1.ln() {
			*self.posterior.lock() = new_posterior;

			Ok(Accept)
		} else {
			Ok(Reject)
		}
	}

	fn finalize(
		&self,
		py: Python,
		operator_index: usize,
		status: StepResult,
	) -> Result<()> {
		self.scheduler.finalize(py, operator_index, status)?;

		if status.is_accept() {
			for likelihood in &self.likelihoods {
				likelihood.get().inner().accept()?;
			}

			for parameter in &self.state {
				py_call_method!(py, parameter, "accept")?;
			}
		} else {
			for likelihood in &self.likelihoods {
				likelihood.get().inner().reject()?;
			}

			for parameter in &self.state {
				py_call_method!(py, parameter, "reject")?;
			}
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
