use anyhow::{Result, ensure};
use pyo3::{intern, prelude::*, types::PyTuple};

use crate::{operator::Proposal, tree::PyTree};
use rng::PyRng;
use util::py_get_attr;

/// Scales a random epoch in a tree
///
/// This parameter is analogous to BEAST2's `ScaleOperator` when it's used on a
/// tree.  It will scale the full tree (so, for now, only its internal nodes,
/// since leaves all have the height of 0).
#[derive(Debug)]
#[pyclass(module = "aspartik.b3.operators", frozen)]
pub struct EpochScale {
	tree: Py<PyTree>,

	/// The scaling ratio will be sampled from `(factor, 1 / factor)`.  So,
	/// the factor must be between 0 and 1 and the smaller it is the larger
	/// the steps will be.
	#[pyo3(get)]
	factor: f64,

	/// Distribution from which the scale is sampled.
	distribution: Py<PyAny>,
	rng: Py<PyRng>,
	#[pyo3(get)]
	weight: f64,
}

#[pymethods]
impl EpochScale {
	#[new]
	fn new(
		tree: Py<PyTree>,
		factor: f64,
		distribution: Py<PyAny>,
		rng: Py<PyRng>,
		weight: f64,
	) -> Result<Self> {
		ensure!(
			0.0 < factor && factor < 1.0,
			"factor must be between 0 and 1, got {factor}"
		);

		Ok(Self {
			tree,
			factor,
			distribution,
			rng,
			weight,
		})
	}

	#[getter]
	fn tree(&self, py: Python) -> Py<PyTree> {
		self.tree.clone_ref(py)
	}

	#[getter]
	fn distribution(&self, py: Python) -> Py<PyAny> {
		self.distribution.clone_ref(py)
	}

	#[getter]
	fn rng(&self, py: Python) -> Py<PyRng> {
		self.rng.clone_ref(py)
	}

	fn __getnewargs__(&self, py: Python) -> PyResult<Py<PyTuple>> {
		(
			self.tree(py),
			self.factor,
			self.distribution(py),
			self.rng(py),
			self.weight,
		)
			.into_pyobject(py)
			.map(|o| o.unbind())
	}

	fn propose(&self, py: Python) -> Result<Proposal> {
		let mut tree = self.tree.get().inner();

		let (low, high) = (self.factor, 1.0 / self.factor);

		let module = PyModule::import(
			py,
			intern!(py, "aspartik.b3.operators._util"),
		)?;
		let func = py_get_attr!(module, "sample_range")?;

		let scale = func
			.call1((
				low,
				high,
				self.distribution(py),
				self.rng(py),
			))?
			.extract::<f64>()?;

		let mut rng = self.rng.get().inner();

		let x = tree.random_internal(&mut rng);
		let y = tree.random_internal(&mut rng);
		let (x_height, y_height) =
			(tree.height_of(&x), tree.height_of(&y));
		let lower = f64::min(x_height, y_height);
		let upper = f64::max(x_height, y_height);

		let move_to = lower + scale * (upper - lower);
		let delta = move_to - upper;

		let mut num_scaled: u32 = 0;

		for node in tree.internals() {
			let height = tree.height_of(&node);
			if lower < height && height <= upper {
				let new_height =
					lower + scale * (height - lower);
				tree.set_height(&node, new_height);
				num_scaled += 1;
			} else if height > upper {
				let new_height = height + delta;
				tree.set_height(&node, new_height);
			}

			if !tree.is_node_height_valid(&node) {
				return Ok(Proposal::Reject());
			}
		}

		if num_scaled < 2 {
			return Ok(Proposal::Reject());
		}

		let ratio = scale.ln() * f64::from(num_scaled);
		Ok(Proposal::Hastings(ratio))
	}
}
