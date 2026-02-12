use anyhow::{Context, Result, anyhow};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::{
	exceptions::PyValueError,
	types::{PyList, PyString, PyType},
};
use rand::distr::{Distribution, weighted::WeightedIndex};
use serde::Serialize;

use std::time::Duration;

use crate::mcmc::StepResult;
use rng::Rng;
use util::{
	py_bail, py_call_method, py_check_method, py_extract_attr,
	py_has_method, time,
};

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize)]
pub enum Proposal {
	Reject(),
	Hastings(f64),
	Accept(),
}

impl<'py> FromPyObject<'_, 'py> for Proposal {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		let py_proposal = obj.extract::<PyProposal>()?;
		Ok(py_proposal.0)
	}
}

impl<'py> IntoPyObject<'py> for Proposal {
	type Target = PyProposal;
	type Output = Bound<'py, PyProposal>;
	type Error = PyErr;

	fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, PyErr> {
		Bound::new(py, PyProposal(self))
	}
}

/// A result of the move proposed by an operator
///
/// While the operators edit the tree directly, they need to communicate the
/// status of their move to `MCMC`.  This is the class used for that.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[pyclass(
	from_py_object,
	module = "aspartik.b3",
	name = "Proposal",
	frozen,
	eq
)]
pub struct PyProposal(Proposal);

#[pymethods]
impl PyProposal {
	/// Aborts the move unconditionally
	///
	/// All of the trees and parameters are rolled back.  This is relatively
	/// fast, as it typically skips recalculating the likelihoods.
	#[classmethod]
	#[pyo3(name = "Reject")]
	fn reject(_cls: Py<PyType>) -> Proposal {
		Proposal::Reject()
	}

	/// Proposes the move with the `ratio`
	///
	/// This is the ratio from the Metropolis–Hastings algorithm.
	#[classmethod]
	#[pyo3(name = "Hastings")]
	fn hastings(_cls: Py<PyType>, ratio: f64) -> Proposal {
		Proposal::Hastings(ratio)
	}

	/// Accepts the move unconditionally
	#[classmethod]
	#[pyo3(name = "Accept")]
	fn accept(_cls: Py<PyType>) -> Proposal {
		Proposal::Accept()
	}

	fn __repr__(&self) -> String {
		match self.0 {
			Proposal::Reject() => "Proposal.Reject()".to_owned(),
			Proposal::Hastings(r) => {
				format!("Proposal.Hastings({r})")
			}
			Proposal::Accept() => "Proposal.Accept()".to_owned(),
		}
	}
}

#[derive(Debug)]
pub struct PyOperator {
	inner: Py<PyAny>,

	has_accept_reject: bool,
}

impl<'py> FromPyObject<'_, 'py> for PyOperator {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "propose");
		py_extract_attr!(obj, "weight", f64)?;

		let has_accept_reject = py_has_method!(obj, "accept")
			&& py_has_method!(obj, "reject");

		Ok(Self {
			inner: obj.to_owned().unbind(),
			has_accept_reject,
		})
	}
}

impl PyOperator {
	pub fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	pub fn propose(&self, py: Python) -> Result<Proposal> {
		let proposal = py_call_method!(py, self.inner, "propose")?;
		let proposal = proposal.extract::<Proposal>(py)?;

		Ok(proposal)
	}

	pub fn repr<'py>(
		&self,
		py: Python<'py>,
	) -> Result<Bound<'py, PyString>> {
		Ok(self.inner.bind(py).repr()?)
	}

	pub fn type_name(&self, py: Python) -> Result<String> {
		Ok(self.inner.bind(py).get_type().name()?.to_string())
	}

	pub fn clone_inner(&self, py: Python) -> Py<PyAny> {
		self.inner.clone_ref(py)
	}

	pub fn accept(&self, py: Python) -> Result<()> {
		if self.has_accept_reject {
			py_call_method!(py, self.inner, "accept")?;
		}
		Ok(())
	}

	pub fn reject(&self, py: Python) -> Result<()> {
		if self.has_accept_reject {
			py_call_method!(py, self.inner, "reject")?;
		}
		Ok(())
	}
}

#[derive(Debug)]
pub struct WeightedScheduler {
	operators: Vec<PyOperator>,
	weights: Vec<f64>,
	statistics: Mutex<Statistics>,
}

type ResultStatistic = [usize; 5];
const EMPTY_RESULT_STATISTIC: ResultStatistic = [0; 5];

#[derive(Debug)]
struct Statistics {
	results: Vec<ResultStatistic>,
	propose: Vec<Duration>,
	likelihood: Vec<Duration>,
}

impl Statistics {
	fn new(len: usize) -> Self {
		Self {
			results: vec![EMPTY_RESULT_STATISTIC; len],
			propose: vec![Duration::default(); len],
			likelihood: vec![Duration::default(); len],
		}
	}
}

impl WeightedScheduler {
	pub fn new(py: Python, operators: Vec<PyOperator>) -> Result<Self> {
		let num_operators = operators.len();

		let mut weights = Vec::with_capacity(num_operators);
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

		Ok(Self {
			operators,
			weights,
			statistics: Statistics::new(num_operators).into(),
		})
	}

	pub fn get_operator(&self, index: usize) -> &PyOperator {
		&self.operators[index]
	}

	pub fn operators(&self, py: Python) -> Vec<Py<PyAny>> {
		self.operators.iter().map(|p| p.clone_inner(py)).collect()
	}

	pub fn random_operator_index(&self, rng: &mut Rng) -> usize {
		let dist = WeightedIndex::new(&self.weights).unwrap();
		dist.sample(rng)
	}

	pub fn finalize(
		&self,
		py: Python,
		index: usize,
		result: StepResult,
	) -> Result<()> {
		if result.is_accept() {
			self.operators[index].accept(py)?;
		} else {
			self.operators[index].reject(py)?;
		}

		let mut statistics = self.statistics.lock();
		statistics.results[index][result.index()] += 1;

		Ok(())
	}

	pub fn make_proposal(
		&self,
		py: Python,
		index: usize,
	) -> Result<Proposal> {
		let operator = &self.operators[index];

		let (proposal, time) = time! {
			operator.propose(py).with_context(|| {
				anyhow!(
					"Operator {} failed while generating a proposal",
					operator.repr(py).unwrap()
				)
			})?
		};

		let mut statistics = self.statistics.lock();
		statistics.propose[index] += time;

		Ok(proposal)
	}

	pub fn record_likelihood_duration(
		&self,
		index: usize,
		duration: Duration,
	) {
		let mut statistics = self.statistics.lock();
		statistics.likelihood[index] += duration;
	}

	pub fn statistics(&self, py: Python) -> Result<Py<PyList>> {
		let statistics = self.statistics.lock();

		let out = PyList::empty(py);

		for i in 0..self.operators.len() {
			let copy = self.operators[i].inner.clone_ref(py);
			let results = statistics.results[i];
			let propose = statistics.propose[i];
			let likelihood = statistics.likelihood[i];

			let tuple = (copy, results, propose, likelihood)
				.into_pyobject(py)?;

			out.append(tuple)?;
		}

		Ok(out.unbind())
	}
}
