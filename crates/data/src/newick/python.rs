use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;

use super::{Tree, parse};

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
	fn new(newick: Option<&str>) -> Result<Self> {
		let tree = match newick {
			None => Tree::new(),
			Some(input) => parse(input)?,
		};

		Ok(PyTree {
			inner: Mutex::new(tree),
		})
	}

	fn __str__(&self) -> String {
		self.inner().into_string()
	}
}
