use std::{fs::File, io::BufReader, path::PathBuf};

use anyhow::Result;
use pyo3::prelude::*;

use crate::tree::python::PyBinaryTree;
use crate::{nexus::read_binary_trees, tree::BinaryTree};

fn to_py_trees(
	py: Python<'_>,
	trees: Vec<BinaryTree>,
) -> Result<Vec<Py<PyBinaryTree>>> {
	let out = trees
		.into_iter()
		.map(|tree| Py::new(py, PyBinaryTree::from(tree)))
		.collect::<PyResult<Vec<_>>>()?;
	Ok(out)
}

#[pyfunction(name = "read_binary_trees_path")]
pub fn py_read_binary_trees_path(
	py: Python<'_>,
	path: PathBuf,
) -> Result<Vec<Py<PyBinaryTree>>> {
	let trees = py.detach(move || {
		let reader = BufReader::new(File::open(path)?);
		read_binary_trees(reader)
	})?;
	to_py_trees(py, trees)
}

#[pyfunction(name = "read_binary_tree_str")]
pub fn py_read_binary_tree_str(
	py: Python<'_>,
	s: &str,
) -> Result<Vec<Py<PyBinaryTree>>> {
	let trees =
		py.detach(move || read_binary_trees(std::io::Cursor::new(s)))?;
	to_py_trees(py, trees)
}
