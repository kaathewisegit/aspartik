use anyhow::Result;
use pyo3::prelude::*;

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
	parameter::{Parameter, PyParameter},
	tree::PyTree,
};
use rng::PyRng;

#[derive(Debug)]
pub struct State {
	/// TODO: parameter serialization
	backup_params: Vec<Parameter>,
	/// Current set of parameters by name.
	params: Vec<PyParameter>,

	pub(crate) tree: PyTree,

	/// Current likelihood, for caching purposes.
	pub(crate) likelihood: f64,

	pub(crate) rng: Py<PyRng>,
}

impl State {
	pub fn new(
		tree: PyTree,
		params: Vec<PyParameter>,
		rng: Py<PyRng>,
	) -> Result<Self> {
		let mut backup_params = Vec::with_capacity(params.len());
		for param in &params {
			backup_params.push(param.inner()?.clone());
		}

		Ok(Self {
			backup_params,
			params,
			tree,
			likelihood: f64::NEG_INFINITY,
			rng,
		})
	}

	/// Accept the current proposal
	pub fn accept(&mut self) -> Result<()> {
		for i in 0..self.params.len() {
			self.backup_params[i] = self.params[i].inner()?.clone();
		}

		self.tree.inner().accept();

		Ok(())
	}

	pub fn reject(&mut self) -> Result<()> {
		for i in 0..self.params.len() {
			*self.params[i].inner()? =
				self.backup_params[i].clone();
		}

		self.tree.inner().reject();

		Ok(())
	}
}

#[derive(Debug, Clone)]
#[pyclass(name = "State", module = "aspartik.b3", frozen)]
pub struct PyState {
	inner: Arc<Mutex<State>>,
}

impl PyState {
	pub fn inner(&self) -> MutexGuard<State> {
		self.inner.lock().unwrap()
	}
}

#[pymethods]
impl PyState {
	#[new]
	fn new(
		tree: PyTree,
		params: Vec<PyParameter>,
		rng: Py<PyRng>,
	) -> Result<Self> {
		let state = State::new(tree, params, rng)?;

		Ok(Self {
			inner: Arc::new(Mutex::new(state)),
		})
	}

	#[getter]
	fn rng(&self, py: Python) -> Py<PyRng> {
		self.inner().rng.clone_ref(py)
	}
}
