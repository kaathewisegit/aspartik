use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;

use super::Tree;

/// A classical Newick tree
///
/// This tree supports node and edge attributes, but it does not parse them in
/// any way.  It also doesn't support extended Newick which includes
/// hybridization events.
///
/// The constructor takes a single line of Newick notation.
///
/// ```python
/// >>> tree = Tree("(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);")
/// >>> str(tree)
/// '(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);'
/// ```
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
			Some(input) => Tree::parse(input)?,
		};

		Ok(PyTree {
			inner: Mutex::new(tree),
		})
	}

	fn __str__(&self) -> String {
		self.inner().into_string()
	}
}

impl<'py> FromPyObject<'_, 'py> for Tree {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		let py_tree = obj.cast::<PyTree>()?;
		let py_tree = py_tree.get();

		Ok(py_tree.inner().clone())
	}
}
