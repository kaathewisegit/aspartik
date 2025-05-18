use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use rand::Rng as _;

use crate::{
	likelihood::{Likelihood, PyLikelihood},
	operator::{Proposal, PyOperator, WeightedScheduler},
	state::PyState,
	tree::PyTree,
	PyLogger, PyPrior,
};
use rng::PyRng;

#[pyfunction]
// XXX: yes, this is a problem, even without the burnin and the trees.  It
// should be probably rolled into an MCMC class which will handle backups and
// running.
#[expect(clippy::too_many_arguments)]
pub fn run(
	py: Python,
	length: usize,
	state: PyState,
	trees: Vec<PyTree>,
	priors: Vec<PyPrior>,
	operators: Vec<PyOperator>,
	likelihood: PyLikelihood,
	mut loggers: Vec<PyLogger>,
	rng: Py<PyRng>,
) -> Result<()> {
	let likelihood = &mut *likelihood.inner();

	// We acquire the lock for the full duration of the program so that we
	// don't spend time locking and unlocking
	let mut scheduler = WeightedScheduler::new(py, operators)?;

	for index in 0..length {
		step(
			py,
			&state,
			&trees,
			&priors,
			likelihood,
			&mut scheduler,
			&rng,
		)
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
	trees: &[PyTree],
	priors: &[PyPrior],
	likelihood: &mut Likelihood,
	scheduler: &mut WeightedScheduler,
	rng: &Py<PyRng>,
) -> Result<()> {
	let operator = scheduler.select_operator(&mut rng.get().inner());

	let hastings = match operator.propose(py, state).with_context(|| {
		anyhow!(
			"Operator {} failed while generating a proposal",
			operator.repr(py).unwrap()
		)
	})? {
		Proposal::Accept() => {
			accept(state, trees, likelihood)?;
			return Ok(());
		}
		Proposal::Reject() => {
			return Ok(());
		}
		Proposal::Hastings(ratio) => ratio,
	};

	for tree in trees {
		tree.inner().verify()?;
	}

	let mut prior: f64 = 0.0;
	for py_prior in priors {
		prior += py_prior.probability(py, state)?;

		// short-circuit on a rejection by any prior
		if prior == f64::NEG_INFINITY {
			reject(state, trees, likelihood)?;
			return Ok(());
		}
	}

	// calculate tree likelihood
	let new_likelihood = likelihood.propose(py)?;

	let posterior = new_likelihood + prior;

	let ratio = posterior - state.inner().likelihood + hastings;

	let random_0_1 = rng.get().inner().random::<f64>();
	if ratio > random_0_1.ln() {
		state.inner().likelihood = posterior;

		accept(state, trees, likelihood)?;
	} else {
		reject(state, trees, likelihood)?;
	}

	Ok(())
}

fn accept(
	state: &PyState,
	trees: &[PyTree],
	likelihood: &mut Likelihood,
) -> Result<()> {
	state.inner().accept()?;
	for tree in trees {
		tree.inner().accept();
	}
	likelihood.accept();

	Ok(())
}

fn reject(
	state: &PyState,
	trees: &[PyTree],
	likelihood: &mut Likelihood,
) -> Result<()> {
	state.inner().reject()?;
	for tree in trees {
		tree.inner().reject();
	}
	likelihood.reject();

	Ok(())
}
