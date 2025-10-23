use std::time::Duration;

use anyhow::Result;
use log::{debug, trace};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::{
	exceptions::PyValueError,
	types::{PyList, PyString, PyType},
};
use rand::distr::{Distribution, weighted::WeightedIndex};

use profiler::profile;
use rng::Rng;
use util::{py_bail, py_call_method, py_check_method, py_extract_attr};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Proposal {
	Reject(),
	Hastings(f64),
	Accept(),
}

impl<'py> FromPyObject<'py> for Proposal {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
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

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[pyclass(module = "aspartik.b3", name = "Proposal", frozen, eq)]
pub struct PyProposal(Proposal);

#[pymethods]
impl PyProposal {
	#[classmethod]
	#[pyo3(name = "Reject")]
	fn reject(_cls: Py<PyType>) -> Proposal {
		Proposal::Reject()
	}

	#[classmethod]
	#[pyo3(name = "Hastings")]
	fn hastings(_cls: Py<PyType>, ratio: f64) -> Proposal {
		Proposal::Hastings(ratio)
	}

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
}

impl<'py> FromPyObject<'py> for PyOperator {
	fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "propose");
		py_extract_attr!(obj, "weight", f64)?;

		let out = Self {
			inner: obj.clone().unbind(),
		};
		debug!(
			target: "b3::operator::extract_bound",
			repr:% = obj.repr()?, id = out.id();
			""
		);
		Ok(out)
	}
}

impl PyOperator {
	pub fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	pub fn propose(&self, py: Python) -> Result<Proposal> {
		let proposal = profile!(
			target: "b3::operator::propose"
			id = self.id();
			py_call_method!(py, self.inner, "propose")?
		);
		let proposal = proposal.extract::<Proposal>(py)?;
		trace!(target: "b3::operator", propose:? = proposal; "");

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

	pub fn accept(&self, _py: Python) -> Result<()> {
		Ok(())
	}

	pub fn reject(&self, _py: Python) -> Result<()> {
		Ok(())
	}
}

#[derive(Debug)]
pub struct WeightedScheduler {
	operators: Vec<PyOperator>,
	weights: Vec<f64>,
	statistics: Mutex<Statistics>,
}

#[derive(Debug)]
struct Statistics {
	accepts: Vec<usize>,
	rejects: Vec<usize>,
	propose: Vec<Duration>,
	likelihood: Vec<Duration>,
}

impl Statistics {
	fn new(len: usize) -> Self {
		Self {
			accepts: vec![0; len],
			rejects: vec![0; len],
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

	pub fn accept(&self, py: Python, index: usize) -> Result<()> {
		self.operators[index].accept(py)?;
		let mut statistics = self.statistics.lock();
		statistics.accepts[index] += 1;
		Ok(())
	}

	pub fn reject(&self, py: Python, index: usize) -> Result<()> {
		self.operators[index].reject(py)?;
		let mut statistics = self.statistics.lock();
		statistics.rejects[index] += 1;
		Ok(())
	}

	pub fn record_propose_duration(
		&self,
		index: usize,
		duration: Duration,
	) {
		let mut statistics = self.statistics.lock();
		statistics.propose[index] += duration;
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
			let accepts = statistics.accepts[i];
			let rejects = statistics.rejects[i];
			let propose = statistics.propose[i];
			let likelihood = statistics.likelihood[i];

			let tuple =
				(copy, accepts, rejects, propose, likelihood)
					.into_pyobject(py)?;

			out.append(tuple)?;
		}

		Ok(out.unbind())
	}
}
