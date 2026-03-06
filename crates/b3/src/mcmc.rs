use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use pyo3::{IntoPyObjectExt, prelude::*, types::PyList};
use rand::RngExt;
use serde::{Deserialize, Serializer, ser::SerializeSeq};

use std::{fs, path::Path};

use crate::{
	PyCallback, PyPrior,
	likelihood::PyLikelihood,
	operator::{Proposal, PyOperator, WeightedScheduler},
	parameters::PyParameter,
};
use rng::{PyRng, Rng};
use util::{seconds_since_unix, time};

/// The main object which runs the analysis
#[pyclass(name = "MCMC", module = "aspartik.b3", frozen)]
pub struct Mcmc {
	posterior: Mutex<f64>,

	current_step: Mutex<usize>,

	state: Vec<PyParameter>,
	priors: Vec<PyPrior>,
	scheduler: WeightedScheduler,
	likelihood: PyLikelihood,
	callbacks: Vec<PyCallback>,
	#[pyo3(get)]
	rng: Py<PyRng>,
}

#[pymethods]
impl Mcmc {
	#[new]
	#[pyo3(signature = (
		state, priors, operators, likelihood, callbacks, rng,
	))]
	fn new(
		py: Python,

		state: Vec<PyParameter>,
		priors: Vec<PyPrior>,
		operators: Vec<PyOperator>,
		likelihood: PyLikelihood,
		callbacks: Vec<PyCallback>,
		rng: Py<PyRng>,
	) -> Result<Mcmc> {
		let scheduler = WeightedScheduler::new(py, operators)?;

		Ok(Mcmc {
			posterior: Mutex::new(f64::NEG_INFINITY),
			current_step: Mutex::new(0),

			state,
			priors,
			scheduler,
			likelihood,
			callbacks,
			rng,
		})
	}

	/// Index of the current MCMC step
	///
	/// Starts from 0, includes burn-in.
	#[getter]
	pub fn current_step(&self) -> usize {
		*self.current_step.lock()
	}

	/// All priors
	#[getter]
	fn priors(&self, py: Python) -> Vec<Py<PyAny>> {
		self.priors.iter().map(|p| p.clone_ref(py)).collect()
	}

	/// A list of active operators
	#[getter]
	fn operators(&self, py: Python) -> Vec<Py<PyAny>> {
		self.scheduler.operators(py)
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
				self_.finish_run(py)?;
				self_.dump_state_to_file(format!(
					"b3-error-{}.state",
					seconds_since_unix(),
				))?;
				Err(err)
			}
		}
	}

	/// Posterior probability for the last accepted step
	#[getter]
	pub fn posterior(&self) -> f64 {
		*self.posterior.lock()
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
		self.scheduler.statistics(py)
	}

	fn dump_state(&self) -> Result<Vec<u8>> {
		let mut ser = serde_verbatim::Serializer::new(Vec::new());
		let mut scratch = Vec::new();

		ser.serialize_u64(*self.current_step.lock() as u64)?;
		ser.serialize_f64(*self.posterior.lock())?;

		let mut seq = ser.serialize_seq(Some(self.state.len()))?;
		for param in &self.state {
			let param = &*param.as_ref();
			param.dump(&mut scratch)?;
			seq.serialize_element(&scratch)?;
			scratch.clear();
		}
		seq.end()?;

		let bytes = self.rng.get().dump()?;
		ser.serialize_bytes(&bytes)?;

		Ok(ser.into_inner())
	}

	fn load_state(&self, bytes: &[u8]) -> Result<()> {
		#[derive(Deserialize)]
		struct De<'a> {
			current_step: u64,
			posterior: f64,
			#[serde(borrow)]
			params: Vec<&'a [u8]>,
			rng: Rng,
		}
		let mut deserializer = serde_verbatim::Deserializer::new(bytes);
		let de = De::deserialize(&mut deserializer)?;

		*self.current_step.lock() = de.current_step as usize;
		*self.posterior.lock() = de.posterior;

		for (i, bytes) in de.params.iter().enumerate() {
			let param = &mut *self.state[i].as_ref();
			param.load(bytes)?;
		}

		*self.rng.get().inner() = de.rng;

		self.likelihood.propose()?;
		self.likelihood.likelihood()?;
		self.likelihood.accept()?;

		Ok(())
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
	fn try_run(this: Py<Self>, py: Python, n: usize) -> Result<()> {
		let self_ = this.get();
		let end = self_.current_step() + n;
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

			self_.finalize_step(py, operator_index, result)?;

			Self::call_callbacks(
				this.clone_ref(py),
				py,
				current_step,
			)?;

			if current_step == end {
				break;
			}
			*self_.current_step.lock() += 1;
		}

		self_.finish_run(py)?;

		Ok(())
	}

	fn dump_state_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
		let state = self.dump_state()?;
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
			self.likelihood.propose()?;
			self.likelihood.likelihood()?
		}};

		if likelihood == f64::NEG_INFINITY {
			bail!("Tree likelihood underflowed");
		}

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

	fn finalize_step(
		&self,
		py: Python,
		operator_index: usize,
		status: StepResult,
	) -> Result<()> {
		self.scheduler.finalize(py, operator_index, status)?;

		if status.is_accept() {
			for parameter in &self.state {
				let parameter = &mut *parameter.as_ref();
				parameter.accept();
			}

			for prior in &self.priors {
				prior.accept(py)?;
			}

			self.likelihood.accept()?;
		} else {
			for parameter in &self.state {
				let parameter = &mut *parameter.as_ref();
				parameter.reject();
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

	fn finish_run(&self, py: Python) -> Result<()> {
		for callback in &self.callbacks {
			callback.finish(py)?;
		}
		Ok(())
	}
}
