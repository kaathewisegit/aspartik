use anyhow::Result;
use pyo3::prelude::*;
use pyo3::{
	exceptions::{PyTypeError, PyValueError},
	types::PyString,
};
use rand::distr::{weighted::WeightedIndex, Distribution};
use tracing::{instrument, trace};

use rng::Rng;
use util::{py_bail, py_call_method};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Proposal {
	Reject(),
	Hastings(f64),
	Accept(),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[pyclass(module = "aspartik.b3", name = "Proposal", frozen)]
pub struct PyProposal(Proposal);

#[pymethods]
impl PyProposal {
	#[staticmethod]
	#[pyo3(name = "Reject")]
	fn reject() -> PyProposal {
		PyProposal(Proposal::Reject())
	}

	#[staticmethod]
	#[pyo3(name = "Hastings")]
	fn hastings(ratio: f64) -> PyProposal {
		PyProposal(Proposal::Hastings(ratio))
	}

	#[staticmethod]
	#[pyo3(name = "Accept")]
	fn accept() -> PyProposal {
		PyProposal(Proposal::Accept())
	}
}

#[derive(Debug)]
pub struct PyOperator {
	inner: PyObject,
}

impl<'py> FromPyObject<'py> for PyOperator {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		let repr = obj.repr()?;
		if !obj.getattr("propose").is_ok_and(|a| a.is_callable()) {
			py_bail!(
				PyTypeError,
				"Operator objects must have a `propose` method, which takes no arguments and returns a `Proposal`.  Got {repr}",
			);
		}

		if obj.getattr("weight")?.extract::<f64>().is_err() {
			py_bail!(
				PyTypeError,
				"Operator must have a `weight` attribute which returns a real number.  Got {repr}",
			);
		}

		let out = Self {
			inner: obj.clone().unbind(),
		};
		trace!(%repr, id = out.id(), "new PyOperator");
		Ok(out)
	}
}

impl PyOperator {
	fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	#[instrument(level = "trace", skip_all, fields(id = self.id()))]
	pub fn propose(&self, py: Python) -> Result<Proposal> {
		let proposal = py_call_method!(py, self.inner, "propose")?;
		let proposal = proposal.extract::<PyProposal>(py)?;
		let proposal = proposal.0;
		trace!(?proposal);

		Ok(proposal)
	}

	pub fn repr<'py>(
		&self,
		py: Python<'py>,
	) -> Result<Bound<'py, PyString>> {
		Ok(self.inner.bind(py).repr()?)
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
			// tries don't need context because they are already
			// checked by PyOperator's `extract_bound`
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

	#[instrument(level = "trace", skip_all)]
	pub fn select_operator(&self, rng: &mut Rng) -> &PyOperator {
		// error handling or validation in `new`
		let dist = WeightedIndex::new(&self.weights).unwrap();

		let index = dist.sample(rng);
		trace!(index);

		&self.operators[index]
	}
}
