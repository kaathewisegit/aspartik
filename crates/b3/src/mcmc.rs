use anyhow::{Context, Result, anyhow, bail, ensure};
use parking_lot::Mutex;
use pyo3::{IntoPyObjectExt, prelude::*, types::PyList};
use rand::RngExt;

use std::{fs, io::Write, path::Path};

use crate::{
	PyCallback, STATE_PROTOCOL_VERSION,
	likelihood::Likelihood,
	operators::{Proposal, PyOperator, Scheduler},
	parameters::PyParameter,
	priors::PyPrior,
};
use rng::PyRng;
use util::{atomic::MonotonicF64, seconds_since_unix, time};

/// The main object which runs the analysis
#[pyclass(name = "MCMC", module = "aspartik.b3", frozen)]
pub struct Mcmc {
	posterior: MonotonicF64,

	current_step: Mutex<usize>,

	priors: Vec<PyPrior>,
	scheduler: Scheduler,
	likelihood: Likelihood,
	callbacks: Vec<PyCallback>,
	#[pyo3(get)]
	rng: Py<PyRng>,

	parameters: Vec<PyParameter>,
}

#[pymethods]
impl Mcmc {
	#[new]
	#[pyo3(signature = (
		priors, operators, likelihood, callbacks, rng,
		optimization_cutoff = 1_000_000,
	))]
	fn new(
		py: Python,

		priors: Vec<PyPrior>,
		operators: Vec<PyOperator>,
		likelihood: Likelihood,
		callbacks: Vec<PyCallback>,
		rng: Py<PyRng>,

		optimization_cutoff: usize,
	) -> Result<Mcmc> {
		let scheduler =
			Scheduler::new(py, operators, optimization_cutoff)?;
		let parameters = scheduler.parameters(py)?;

		let out = Mcmc {
			posterior: f64::NEG_INFINITY.into(),
			current_step: Mutex::new(0),

			priors,
			scheduler,
			likelihood,
			callbacks,
			rng,
			parameters,
		};
		out.posterior
			.store(out.likelihood.likelihood()? + out.prior(py)?);
		Ok(out)
	}

	/// Index of the current MCMC step
	///
	/// Starts from 0, includes burn-in.
	#[getter]
	pub fn current_step(&self) -> usize {
		*self.current_step.lock()
	}

	#[getter]
	fn parameters(&self, py: Python) -> Vec<Py<PyAny>> {
		self.parameters.iter().map(|p| p.into_py_any(py)).collect()
	}

	/// All priors
	#[getter]
	fn priors(&self, py: Python) -> Vec<Py<PyAny>> {
		self.priors.iter().map(|p| p.clone_ref(py)).collect()
	}

	/// A list of active operators
	#[getter]
	fn operators(&self, py: Python) -> Vec<Py<PyAny>> {
		self.scheduler.py_operators(py)
	}

	#[getter]
	fn likelihood(&self, py: Python) -> PyResult<Py<PyAny>> {
		self.likelihood.clone_ref(py).into_py_any(py)
	}

	/// A list of callbacks
	#[getter]
	fn callbacks(&self, py: Python) -> Vec<Py<PyAny>> {
		self.callbacks.iter().map(|l| l.clone_ref(py)).collect()
	}

	/// Execute `n` steps of the Markov chain
	///
	/// This yields flow control to the Rust core until the simulation is
	/// done.  Press Ctrl+C to interrupt and stop the execution.
	fn run(this: Py<Self>, py: Python, n: usize) -> Result<()> {
		match Self::try_run(this.clone_ref(py), py, n) {
			Ok(()) => Ok(()),
			Err(err) => {
				let self_ = this.get();
				self_.finish_run(py, this.clone_ref(py))?;
				self_.dump_state_to_file(
					py,
					format!(
						"b3-error-{}.state",
						seconds_since_unix(),
					),
				)?;
				Err(err)
			}
		}
	}

	/// Posterior probability for the last accepted step
	#[getter]
	pub fn posterior(&self) -> f64 {
		self.posterior.load()
	}

	#[getter]
	pub fn likelihood_value(&self) -> Result<f64> {
		self.likelihood.likelihood()
	}

	/// Prior likelihood for the current step
	///
	/// Note that unlike [`posterior`](#MCMC.posterior) and
	/// [`Likelihood`](#MCMC.Likelihood), this property isn't cached.  It
	/// will trigger a recalculation on all priors on each access.
	#[getter]
	pub fn prior(&self, py: Python) -> Result<f64> {
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

	/// Operator statistics for this run
	///
	/// Returns a list of `(operator, results, propose, likelihood)` tuples
	/// for each operator.  `propose` and `likelihood` records the total
	/// time the MCMC spent waiting for the operator to generate a proposal
	/// and calculate it respectively.  `operator` is the reference to the
	/// original operator object.  And `results` is a list of step results.
	#[getter]
	fn operator_statistics(&self, py: Python) -> Result<Py<PyList>> {
		self.scheduler.py_statistics(py)
	}

	fn dump_state(&self, py: Python) -> Result<Vec<u8>> {
		let mut out = Vec::new();
		self.dump(py, &mut out)?;
		Ok(out)
	}

	fn load_state(&self, py: Python, mut bytes: &[u8]) -> Result<()> {
		let version = verbatim::read_u32_le(&mut bytes)?;
		ensure!(
			version == STATE_PROTOCOL_VERSION,
			"Cannot load state: protocol version v{version} is incompatible with the currently supported version v{STATE_PROTOCOL_VERSION}."
		);
		*self.current_step.lock() =
			verbatim::read_u64_le(&mut bytes)? as usize;
		self.posterior.store(verbatim::read_f64_le(&mut bytes)?);

		for param in &self.parameters {
			param.as_dyn().load(&mut bytes)?;
		}

		self.scheduler.load(py, &mut bytes)?;

		self.rng.get().load(&mut bytes)?;

		// update calculators
		self.likelihood.likelihood()?;
		// update priors
		for py_prior in &self.priors {
			py_prior.probability(py)?;
		}

		for parameter in &self.parameters {
			parameter.as_dyn().accept();
		}

		for prior in &self.priors {
			prior.accept(py)?;
		}

		self.likelihood.accept()?;

		Ok(())
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
	/// Operator returned `Proposal::Reject`
	Abort,
	/// A prior returned negative infinity
	PriorReject,
	/// Regular MCMC accept
	Accept,
	/// Regular MCMC reject
	Reject,
}

impl StepResult {
	pub fn is_accept(&self) -> bool {
		*self == Self::Accept
	}

	pub fn is_reject(&self) -> bool {
		!self.is_accept()
	}
}

impl Mcmc {
	pub fn scheduler(&self) -> &Scheduler {
		&self.scheduler
	}

	fn dump(&self, py: Python, writer: &mut dyn Write) -> Result<()> {
		verbatim::write_u32_le(writer, STATE_PROTOCOL_VERSION)?;
		verbatim::write_u64_le(
			writer,
			*self.current_step.lock() as u64,
		)?;
		verbatim::write_f64_le(writer, self.posterior.load())?;

		for parameter in &self.parameters {
			parameter.as_dyn().dump(writer)?;
		}

		self.scheduler.dump(py, writer)?;

		self.rng.get().dump(writer)
	}

	fn try_run(this: Py<Self>, py: Python, n: usize) -> Result<()> {
		let self_ = this.get();
		for _ in 0..n {
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

			self_.finalize_step(py, operator_index, result)?;

			self_.scheduler.tune(py, current_step)?;

			Self::call_callbacks(
				this.clone_ref(py),
				py,
				current_step,
			)?;

			*self_.current_step.lock() += 1;
		}

		self_.finish_run(py, this.clone_ref(py))?;

		Ok(())
	}

	fn dump_state_to_file(
		&self,
		py: Python,
		path: impl AsRef<Path>,
	) -> Result<()> {
		let state = self.dump_state(py)?;
		fs::write(path, state)?;
		Ok(())
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
			Proposal(f64::NEG_INFINITY) => {
				return Ok(Abort);
			}
			Proposal(ratio) => ratio,
		};

		let prior = self.prior(py)?;
		// The proposal will be rejected regardless of likelihood
		if prior == f64::NEG_INFINITY {
			return Ok(PriorReject);
		}

		let (likelihood, time) = time! {self.likelihood.likelihood()?};

		if likelihood == f64::NEG_INFINITY {
			bail!("Tree likelihood underflowed");
		}

		self.scheduler
			.record_likelihood_duration(operator_index, time);

		let new_posterior = likelihood + prior;

		let old_posterior = self.posterior.load();

		let ratio = new_posterior - old_posterior + hastings;

		let random_0_1 = self.rng.get().inner().random::<f64>();
		if ratio > random_0_1.ln() {
			self.posterior.store(new_posterior);

			Ok(Accept)
		} else {
			Ok(Reject)
		}
	}

	fn finalize_step(
		&self,
		py: Python,
		operator_index: usize,
		status: StepResult,
	) -> Result<()> {
		self.scheduler.finalize(py, operator_index, status)?;

		if status.is_accept() {
			for parameter in &self.parameters {
				parameter.as_dyn().accept();
			}

			for prior in &self.priors {
				prior.accept(py)?;
			}

			self.likelihood.accept()?;
		} else {
			for parameter in &self.parameters {
				parameter.as_dyn().reject();
			}

			for prior in &self.priors {
				prior.reject(py)?;
			}

			self.likelihood.reject()?;
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

	fn finish_run(&self, py: Python, mcmc: Py<Mcmc>) -> Result<()> {
		for callback in &self.callbacks {
			callback.finish(py, mcmc.clone_ref(py))?;
		}
		Ok(())
	}
}
