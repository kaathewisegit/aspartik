#![expect(unused)]

use anyhow::Result;
use parking_lot::Mutex;
use pyo3::prelude::*;

use super::{PyCpu4Likelihood, PyLikelihood, PyParallel4Likelihood};
use crate::{likelihood, parameters::PyClassVector};

enum ConcreteLikelihood {
	Cpu(Py<PyCpu4Likelihood>),
	Parallel(Py<PyParallel4Likelihood>),
}

impl<'py> FromPyObject<'_, 'py> for ConcreteLikelihood {
	type Error = PyErr;

	fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
		if let Ok(l) = obj.cast::<PyCpu4Likelihood>() {
			Ok(Self::Cpu(l.into()))
		} else if let Ok(l) = obj.cast::<PyParallel4Likelihood>() {
			Ok(Self::Parallel(l.into()))
		} else {
			todo!("descriptive error")
		}
	}
}

impl ConcreteLikelihood {
	fn propose(&self, py: Python) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().propose(py),
			Self::Parallel(l) => l.get().propose(py),
		}
	}

	fn likelihood(&self, py: Python) -> Result<f64> {
		match self {
			Self::Cpu(l) => l.get().likelihood(),
			Self::Parallel(l) => l.get().likelihood(),
		}
	}

	fn accept(&self, py: Python) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().accept(),
			Self::Parallel(l) => l.get().accept(),
		}
	}

	fn reject(&self, py: Python) -> Result<()> {
		match self {
			Self::Cpu(l) => l.get().reject(),
			Self::Parallel(l) => l.get().reject(),
		}
	}

	fn num_patterns(&self) -> usize {
		match self {
			Self::Cpu(l) => l.get().num_patterns(),
			Self::Parallel(l) => l.get().num_patterns(),
		}
	}

	fn pattern_likelihoods(&self) -> Result<Vec<f64>> {
		match self {
			Self::Cpu(l) => l.get().pattern_likelihoods(),
			Self::Parallel(l) => l.get().pattern_likelihoods(),
		}
	}
}

pub struct HeteroLikelihood {
	likelihoods: Vec<ConcreteLikelihood>,
	classes: Py<PyClassVector>,
}

impl HeteroLikelihood {
	fn new(
		py: Python,
		likelihoods: Vec<ConcreteLikelihood>,
	) -> Result<Self> {
		let num_classes = likelihoods.len() as u8;
		let num_patterns = likelihoods[0].num_patterns();
		let classes = Py::new(
			py,
			PyClassVector::new(num_classes, num_patterns),
		)?;

		Ok(Self {
			likelihoods,
			classes,
		})
	}

	fn propose(&self, py: Python) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.propose(py)?;
		}
		Ok(())
	}

	fn likelihood(&self) -> Result<f64> {
		let likelihoods: Vec<Vec<f64>> = self
			.likelihoods
			.iter()
			.map(|l| l.pattern_likelihoods())
			.collect::<Result<_>>()?;

		let classes = &*self.classes.get().inner();

		let mut out = 0.0;

		for i in 0..classes.len() {
			let class = classes[i] as usize;
			out += likelihoods[class][i];
		}

		Ok(out)
	}

	fn accept(&self, py: Python) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.accept(py)?;
		}
		Ok(())
	}

	fn reject(&self, py: Python) -> Result<()> {
		for likelihood in &self.likelihoods {
			likelihood.reject(py)?;
		}
		Ok(())
	}
}

#[pyclass(
	name = "HeteroLikelihood",
	module = "aspartik.b3.likelihoods",
	frozen
)]
pub struct PyHeteroLikelihood {
	inner: Mutex<HeteroLikelihood>,
}

#[pymethods]
impl PyHeteroLikelihood {
	#[new]
	fn new(
		py: Python,
		likelihoods: Vec<ConcreteLikelihood>,
	) -> Result<Self> {
		Ok(Self {
			inner: Mutex::new(HeteroLikelihood::new(
				py,
				likelihoods,
			)?),
		})
	}

	fn propose(&self, py: Python) -> Result<()> {
		self.inner.lock().propose(py)
	}

	fn likelihood(&self) -> Result<f64> {
		self.inner.lock().likelihood()
	}

	fn accept(&self, py: Python) -> Result<()> {
		self.inner.lock().accept(py)
	}

	fn reject(&self, py: Python) -> Result<()> {
		self.inner.lock().reject(py)
	}

	#[getter]
	fn class_vector(&self, py: Python) -> Py<PyClassVector> {
		self.inner.lock().classes.clone_ref(py)
	}
}
