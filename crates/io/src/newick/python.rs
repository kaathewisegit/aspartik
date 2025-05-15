use pyo3::prelude::*;

use std::sync::{Arc, Mutex, MutexGuard};

use super::{Node, Tree};

#[derive(Debug, Clone)]
#[pyclass(name = "Node", module = "aspartik.io.newick", frozen)]
#[repr(transparent)]
pub(crate) struct PyNode {
	inner: Arc<Mutex<Node>>,
}

impl PyNode {
	fn inner(&self) -> MutexGuard<Node> {
		self.inner.lock().expect("Mutex was poisoned")
	}
}

#[pymethods]
impl PyNode {
	#[new]
	#[pyo3(signature = (name, attributes = None, distance = None))]
	fn new(
		name: String,
		attributes: Option<String>,
		distance: Option<f64>,
	) -> Self {
		let node = Node::new(
			name,
			attributes.unwrap_or_default(),
			distance.expect("TODO take Option in constructor"),
		);
		PyNode {
			inner: Arc::new(Mutex::new(node)),
		}
	}

	#[getter]
	fn name(&self) -> String {
		self.inner().name.clone()
	}

	#[getter]
	fn distance(&self) -> Option<f64> {
		self.inner().distance
	}
}

#[derive(Debug, Clone)]
#[pyclass(name = "Tree", module = "aspartik.io.newick", frozen)]
#[repr(transparent)]
pub(crate) struct PyTree {
	inner: Arc<Mutex<Tree>>,
}

impl PyTree {
	fn inner(&self) -> MutexGuard<Tree> {
		self.inner.lock().expect("Mutex was poisoned")
	}
}

#[pymethods]
impl PyTree {
	#[new]
	fn new() -> Self {
		PyTree {
			inner: Arc::new(Mutex::new(Tree::new())),
		}
	}

	fn __str__(&self) -> String {
		self.inner().serialize()
	}
}
