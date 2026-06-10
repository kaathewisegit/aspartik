use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::{
	exceptions::PyValueError,
	types::{PyList, PyString, PyType},
};
use rand::distr::{Distribution, weighted::WeightedIndex};

use std::{
	collections::HashSet,
	io::Write,
	ops::{Index, IndexMut},
	time::Duration,
};

use crate::{mcmc::StepResult, parameters::PyParameter};
use rng::Rng;
use util::{
	atomic::MonotonicU32, py_bail, py_call_method, py_check_method,
	py_extract_attr, py_has_method, time,
};
use verbatim::{Deserialize, Serialize};

mod classvec_flip;
mod fhspr;
mod subtree_leap;

pub use classvec_flip::ClassvecFlip;
pub use fhspr::FixedHeightSPR;
pub use subtree_leap::SubtreeLeap;

/// A result of the move proposed by an operator
///
/// While the operators edit the tree directly, they need to communicate the
/// status of their move to `MCMC`.  This is the class used for that.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[pyclass(from_py_object, module = "aspartik.b3", frozen, eq)]
pub struct Proposal(pub f64);

impl Proposal {
	pub fn abort() -> Proposal {
		Self(f64::NEG_INFINITY)
	}

	pub fn hastings(ratio: f64) -> Proposal {
		Self(ratio)
	}
}

#[pymethods]
impl Proposal {
	/// Aborts the move unconditionally
	///
	/// All of the trees and parameters are rolled back.  This is relatively
	/// fast, as it typically skips recalculating the likelihoods.
	#[classmethod]
	#[pyo3(name = "Abort")]
	fn py_abort(_cls: Py<PyType>) -> Proposal {
		Self::abort()
	}

	/// Proposes the move with the `ratio`
	///
	/// This is the ratio from the Metropolis–Hastings algorithm.
	#[classmethod]
	#[pyo3(name = "Hastings")]
	fn py_hastings(_cls: Py<PyType>, ratio: f64) -> Proposal {
		Proposal(ratio)
	}

	fn __repr__(&self) -> String {
		match self.0 {
			f64::NEG_INFINITY => "Proposal.Abort()".to_owned(),
			r => {
				format!("Proposal.Hastings({r})")
			}
		}
	}
}

#[derive(Debug)]
pub struct PyOperator {
	inner: Py<PyAny>,

	has_accept_reject: bool,
	has_tuning: bool,
	accepts: MonotonicU32,
	rejects: MonotonicU32,
}

impl<'py> FromPyObject<'_, 'py> for PyOperator {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		py_check_method!(obj, "propose");
		py_check_method!(obj, "parameters");
		py_extract_attr!(obj, "weight", f64)?;

		let has_accept_reject = py_has_method!(obj, "accept")
			&& py_has_method!(obj, "reject");

		let has_tuning = py_has_method!(obj, "set_tuning");

		let out = Self {
			inner: obj.to_owned().unbind(),
			has_accept_reject,
			has_tuning,
			accepts: 0.into(),
			rejects: 0.into(),
		};
		out.set_tuning(obj.py(), 0.75)?;
		Ok(out)
	}
}

impl PyOperator {
	pub fn id(&self) -> usize {
		self.inner.as_ptr() as usize
	}

	pub fn propose(&self, py: Python) -> Result<Proposal> {
		let proposal = py_call_method!(py, self.inner, "propose")?;
		let Ok(proposal) = proposal.extract::<Proposal>(py) else {
			bail!(
				"Expected operator {} to return `Propose` from the `propose` method, got {}",
				self.inner,
				proposal,
			);
		};

		Ok(proposal)
	}

	pub fn parameters(&self, py: Python) -> Result<Vec<PyParameter>> {
		let params = py_call_method!(py, self.inner, "parameters")?;
		let Ok(params) = params.extract::<Vec<PyParameter>>(py) else {
			bail!(
				"Expected operator {} to return a list of parameters from the `parameters` method, got {}",
				self.inner,
				params,
			);
		};
		Ok(params)
	}

	pub fn get_tuning(&self, py: Python) -> Result<Option<f64>> {
		if self.has_tuning {
			Ok(Some(py_call_method!(py, self.inner, "get_tuning")?
				.extract::<f64>(py)?))
		} else {
			Ok(None)
		}
	}

	pub fn set_tuning(&self, py: Python, value: f64) -> Result<()> {
		if self.has_tuning {
			py_call_method!(py, self.inner, "set_tuning", value)?;
		}
		Ok(())
	}

	pub fn tune(&self, py: Python) -> Result<()> {
		let accepts = f64::from(self.accepts.load());
		let rejects = f64::from(self.rejects.load());
		let ratio = accepts / (accepts + rejects);
		if ratio.is_nan() {
			// accepts + rejects = 0 on the firts call, bail
			return Ok(());
		}

		let Some(old_tuning) = self.get_tuning(py)? else {
			return Ok(());
		};

		// This is a somewhat odd optimization routine because it
		// doesn't decrease step size over time.  The idea is very
		// simple: if ratio > 0.234 we decrease the tuning parameter,
		// making the operator more bold, which should decrease
		// acceptance.  And visa-versa.
		//
		// The rate of change (0.05) is tied to how often we tune.
		let new_tuning =
			(old_tuning - 0.05 * (ratio - 0.234)).clamp(0.01, 0.99);

		self.set_tuning(py, new_tuning)?;

		self.accepts.store(0);
		self.rejects.store(0);

		Ok(())
	}

	pub fn load(&self, py: Python, bytes: &mut &[u8]) -> Result<()> {
		self.accepts.store(u32::deserialize(bytes)?);
		self.rejects.store(u32::deserialize(bytes)?);
		if self.has_tuning {
			self.set_tuning(py, f64::deserialize(bytes)?)?;
		}
		Ok(())
	}

	pub fn dump(&self, py: Python, writer: &mut dyn Write) -> Result<()> {
		self.accepts.load().serialize(writer)?;
		self.rejects.load().serialize(writer)?;
		if let Some(tuning) = self.get_tuning(py)? {
			tuning.serialize(writer)?;
		};
		Ok(())
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
		self.accepts.add(1);
		Ok(())
	}

	pub fn reject(&self, py: Python) -> Result<()> {
		if self.has_accept_reject {
			py_call_method!(py, self.inner, "reject")?;
		}
		self.rejects.add(1);
		Ok(())
	}
}

#[derive(Debug)]
pub struct Scheduler {
	operators: Vec<PyOperator>,
	weights: Vec<f64>,
	optimization_cutoff: usize,
	statistics: Mutex<Statistics>,
}

pub type OperatorStatistics = ResultStatistics;

#[derive(Debug, Clone, Copy, Default)]
pub struct ResultStatistics {
	pub aborts: u64,
	pub prior_rejects: u64,
	pub accepts: u64,
	pub rejects: u64,
}

impl Index<StepResult> for ResultStatistics {
	type Output = u64;

	fn index(&self, result: StepResult) -> &u64 {
		match result {
			StepResult::Abort => &self.aborts,
			StepResult::PriorReject => &self.prior_rejects,
			StepResult::Accept => &self.accepts,
			StepResult::Reject => &self.rejects,
		}
	}
}

impl IndexMut<StepResult> for ResultStatistics {
	fn index_mut(&mut self, result: StepResult) -> &mut u64 {
		match result {
			StepResult::Abort => &mut self.aborts,
			StepResult::PriorReject => &mut self.prior_rejects,
			StepResult::Accept => &mut self.accepts,
			StepResult::Reject => &mut self.rejects,
		}
	}
}

#[derive(Debug)]
struct Statistics {
	results: Vec<OperatorStatistics>,
	propose: Vec<Duration>,
	likelihood: Vec<Duration>,
}

impl Statistics {
	fn new(len: usize) -> Self {
		Self {
			results: vec![ResultStatistics::default(); len],
			propose: vec![Duration::default(); len],
			likelihood: vec![Duration::default(); len],
		}
	}
}

impl Scheduler {
	pub fn new(
		py: Python,
		operators: Vec<PyOperator>,
		optimization_cutoff: usize,
	) -> Result<Self> {
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
			optimization_cutoff,
			statistics: Statistics::new(num_operators).into(),
		})
	}

	pub fn parameters(&self, py: Python) -> Result<Vec<PyParameter>> {
		let mut out = Vec::new();
		let mut set = HashSet::new();

		for operator in &self.operators {
			for parameter in operator.parameters(py)? {
				if set.insert(parameter.as_ptr()) {
					out.push(parameter);
				}
			}
		}

		Ok(out)
	}

	pub fn get_operator(&self, index: usize) -> &PyOperator {
		&self.operators[index]
	}

	pub fn operators(&self) -> &[PyOperator] {
		&self.operators
	}

	pub fn py_operators(&self, py: Python) -> Vec<Py<PyAny>> {
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
		statistics.results[index][result] += 1;

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

	pub fn statistics(&self) -> Vec<OperatorStatistics> {
		self.statistics.lock().results.clone()
	}

	pub fn py_statistics(&self, py: Python) -> Result<Py<PyList>> {
		let statistics = self.statistics.lock();

		let out = PyList::empty(py);

		for i in 0..self.operators.len() {
			let copy = self.operators[i].inner.clone_ref(py);
			let results = &statistics.results[i];
			let propose = statistics.propose[i];
			let likelihood = statistics.likelihood[i];

			let tuple = (
				copy,
				(
					results.aborts,
					results.prior_rejects,
					results.accepts,
					results.rejects,
				),
				propose,
				likelihood,
			)
				.into_pyobject(py)?;

			out.append(tuple)?;
		}

		Ok(out.unbind())
	}

	pub fn tune(&self, py: Python, step: usize) -> Result<()> {
		if step > self.optimization_cutoff {
			return Ok(());
		}

		for operator in &self.operators {
			operator.tune(py)?;
		}
		Ok(())
	}

	pub fn load(&self, py: Python, bytes: &mut &[u8]) -> Result<()> {
		for operator in &self.operators {
			operator.load(py, bytes)?;
		}
		Ok(())
	}

	pub fn dump(&self, py: Python, writer: &mut dyn Write) -> Result<()> {
		for operator in &self.operators {
			operator.dump(py, writer)?;
		}
		Ok(())
	}
}
