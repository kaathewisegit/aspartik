use pyo3::prelude::*;

use std::sync::{Arc, Mutex, MutexGuard};

use super::{Node, Tree};

#[derive(Debug, Clone)]
#[pyclass(name = "Node", module = "aspartik.data.newick", frozen)]
#[repr(transparent)]
pub(crate) struct PyNode {
	inner: Arc<Mutex<Node>>,
}

impl PyNode {
	fn inner(&self) -> MutexGuard<'_, Node> {
		self.inner.lock().expect("Mutex was poisoned")
	}
}

#[pymethods]
impl PyNode {
	#[new]
	#[pyo3(signature = (name, attributes = None))]
	fn new(name: String, attributes: Option<String>) -> Self {
		let node = Node::new(name, attributes.unwrap_or_default());
		PyNode {
			inner: Arc::new(Mutex::new(node)),
		}
	}

	#[getter]
	fn name(&self) -> String {
		self.inner().name.clone()
	}
}

#[derive(Debug, Clone)]
#[pyclass(name = "Tree", module = "aspartik.data.newick", frozen)]
#[repr(transparent)]
pub(crate) struct PyTree {
	inner: Arc<Mutex<Tree>>,
}

#[expect(unused)]
impl PyTree {
	fn inner(&self) -> MutexGuard<'_, Tree> {
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
		todo!()
		// self.inner().serialize()
	}
}
