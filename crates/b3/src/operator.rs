use anyhow::Result;
use log::{debug, trace};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::{
	exceptions::{PyTypeError, PyValueError},
	types::{PyString, PyType},
};
use rand::distr::{weighted::WeightedIndex, Distribution};

use profiler::profile;
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

impl From<Proposal> for PyProposal {
	fn from(value: Proposal) -> PyProposal {
		PyProposal(value)
	}
}

#[pymethods]
impl PyProposal {
	#[classmethod]
	#[pyo3(name = "Reject")]
	fn reject(_cls: Py<PyType>) -> PyProposal {
		PyProposal(Proposal::Reject())
	}

	#[classmethod]
	#[pyo3(name = "Hastings")]
	fn hastings(_cls: Py<PyType>, ratio: f64) -> PyProposal {
		PyProposal(Proposal::Hastings(ratio))
	}

	#[classmethod]
	#[pyo3(name = "Accept")]
	fn accept(_cls: Py<PyType>) -> PyProposal {
		PyProposal(Proposal::Accept())
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
	inner: PyObject,
	accepts: Mutex<usize>,
	rejects: Mutex<usize>,
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
			accepts: Mutex::new(0),
			rejects: Mutex::new(0),
		};
		debug!(
			target: "b3::operator::extract_bound",
			repr:%, id = out.id();
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
		let proposal = proposal.extract::<PyProposal>(py)?;
		let proposal = proposal.0;
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

	pub fn accept(&self) {
		*self.accepts.lock() += 1;
	}

	pub fn reject(&self) {
		*self.rejects.lock() += 1;
	}
}

#[derive(Debug)]
pub struct WeightedScheduler {
	operators: Vec<PyOperator>,
	weights: Vec<f64>,
	current: Mutex<usize>,
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

		Ok(Self {
			operators,
			weights,
			current: Mutex::new(0),
		})
	}

	pub fn select_operator(&self, rng: &mut Rng) -> &PyOperator {
		// error handling or validation in `new`
		let dist = WeightedIndex::new(&self.weights).unwrap();

		let index = dist.sample(rng);
		*self.current.lock() = index;

		trace!(
			target: "b3::operators::select_operator",
			index;
			""
		);

		&self.operators[index]
	}

	pub fn accept(&self) {
		self.operators[*self.current.lock()].accept();
	}

	pub fn reject(&self) {
		self.operators[*self.current.lock()].reject();
	}

	pub fn report(&self, py: Python) -> Result<()> {
		println!(
			"{: <20}{: <12}{: <12}",
			"Operator", "#accept", "#reject"
		);
		for operator in &self.operators {
			let name = operator.type_name(py)?;
			let accepts = *operator.accepts.lock();
			let rejects = *operator.rejects.lock();
			println!(
				"{: <20}{: <12}{: <12}",
				name, accepts, rejects
			);
		}
		Ok(())
	}
}
