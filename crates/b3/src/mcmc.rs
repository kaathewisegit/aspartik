use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use rand::Rng as _;

use crate::{
	likelihood::{Likelihood, PyLikelihood},
	operator::{Proposal, PyOperator, WeightedScheduler},
	state::PyState,
	PyLogger, PyPrior,
};

#[pyfunction]
pub fn run(
	py: Python,
	length: usize,
	state: PyState,
	priors: Vec<PyPrior>,
	operators: Vec<PyOperator>,
	likelihood: PyLikelihood,
	mut loggers: Vec<PyLogger>,
) -> Result<()> {
	let likelihood = &mut *likelihood.inner();

	// We acquire the lock for the full duration of the program so that we
	// don't spend time locking and unlocking
	let mut scheduler = WeightedScheduler::new(py, operators)?;

	for index in 0..length {
		step(py, &state, &priors, likelihood, &mut scheduler)
			.with_context(|| anyhow!("Failed on step {index}"))?;

		for logger in &mut loggers {
			logger.log(py, state.clone(), index).with_context(
				|| anyhow!("Failed to log on step {index}"),
			)?;
		}
	}

	Ok(())
}

fn step(
	py: Python,
	state: &PyState,
	priors: &[PyPrior],
	likelihood: &mut Likelihood,
	scheduler: &mut WeightedScheduler,
) -> Result<()> {
	let operator =
		scheduler.select_operator(&mut state.inner().rng.get().inner());

	let hastings = match operator.propose(py, state).with_context(|| {
		anyhow!(
			"Operator {} failed while generating a proposal",
			operator.repr(py).unwrap()
		)
	})? {
		Proposal::Accept() => {
			accept(state, likelihood)?;
			return Ok(());
		}
		Proposal::Reject() => {
			return Ok(());
		}
		Proposal::Hastings(ratio) => ratio,
	};

	state.inner().tree.inner().verify()?;

	let mut prior: f64 = 0.0;
	for py_prior in priors {
		prior += py_prior.probability(py, state)?;

		// short-circuit on a rejection by any prior
		if prior == f64::NEG_INFINITY {
			reject(state, likelihood)?;
			return Ok(());
		}
	}

	// calculate tree likelihood
	let new_likelihood = likelihood.propose(py, state)?;

	let posterior = new_likelihood + prior;

	let ratio = posterior - state.inner().likelihood + hastings;

	let random_0_1 = state.inner().rng.get().inner().random::<f64>();
	if ratio > random_0_1.ln() {
		state.inner().likelihood = posterior;

		accept(state, likelihood)?;
	} else {
		reject(state, likelihood)?;
	}

	Ok(())
}

fn accept(state: &PyState, likelihood: &mut Likelihood) -> Result<()> {
	state.inner().accept()?;
	likelihood.accept();

	Ok(())
}

fn reject(state: &PyState, likelihood: &mut Likelihood) -> Result<()> {
	state.inner().reject()?;
	likelihood.reject();

	Ok(())
}
