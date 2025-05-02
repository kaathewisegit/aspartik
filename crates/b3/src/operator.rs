use anyhow::Result;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use rand::distr::{weighted::WeightedIndex, Distribution};

use crate::state::PyState;
use rng::Rng;
use util::{py_bail, py_call_method};

#[derive(Debug, Clone)]
#[pyclass(frozen)]
pub enum Proposal {
	Accept(),
	Reject(),
	Hastings(f64),
}

#[derive(Debug)]
pub struct PyOperator {
	inner: PyObject,
}

impl<'py> FromPyObject<'py> for PyOperator {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		if !obj.getattr("propose")?.is_callable() {
			return Err(PyTypeError::new_err("Operator objects must have a `propose` method, which takes `State` and returns a `Proposal`"));
		}

		if obj.getattr("weight")?.extract::<f64>().is_err() {
			return Err(PyTypeError::new_err("Operator must have a `weight` attribute which returns a real number"));
		}

		Ok(Self {
			inner: obj.clone().unbind(),
		})
	}
}

impl PyOperator {
	pub fn propose(&self, py: Python, state: &PyState) -> Result<Proposal> {
		let args = (state.clone(),);
		let proposal =
			py_call_method!(py, self.inner, "propose", args)?;
		let proposal = proposal.extract::<Proposal>(py)?;

		Ok(proposal)
	}
}

#[derive(Debug)]
pub struct WeightedScheduler {
	operators: Vec<PyOperator>,
	weights: Vec<f64>,
}

impl WeightedScheduler {
	pub fn new(py: Python, operators: Vec<PyOperator>) -> Result<Self> {
		let mut weights = vec![];
		for operator in &operators {
			let weight = operator
				.inner
				.getattr(py, "weight")?
				.extract::<f64>(py)?;
			weights.push(weight);
		}

		if operators.is_empty() {
			py_bail!(
				PyValueError,
				"Operator list must not be empty",
			);
		}

		Ok(Self { operators, weights })
	}

	pub fn select_operator(&mut self, rng: &mut Rng) -> &PyOperator {
		// error handling or validation in `new`
		let dist = WeightedIndex::new(&self.weights).unwrap();

		let index = dist.sample(rng);

		&self.operators[index]
	}
}
