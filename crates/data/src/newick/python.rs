use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;

use super::Tree;

#[derive(Debug)]
#[pyclass(name = "Tree", module = "aspartik.data.newick", frozen)]
#[repr(transparent)]
pub(crate) struct PyTree {
	inner: Mutex<Tree>,
}

impl PyTree {
	fn inner(&self) -> MutexGuard<'_, Tree> {
		self.inner.lock()
	}
}

#[pymethods]
impl PyTree {
	#[new]
	#[pyo3(signature = (newick = None))]
	fn new(newick: Option<&str>) -> Self {
		let tree = match newick {
			None => Tree::new(),
			Some(_) => todo!("parse the tree"),
		};

		PyTree {
			inner: Mutex::new(tree),
		}
	}

	fn __str__(&self) -> String {
		self.inner().into_string()
	}
}
