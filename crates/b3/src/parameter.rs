use anyhow::{ensure, Result};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::{
	class::basic::CompareOp,
	conversion::FromPyObjectBound,
	exceptions::{PyIndexError, PyTypeError, PyValueError},
	types::PyTuple,
};
use serde::{Deserialize, Serialize};

use std::fmt::{self, Display};

use skvec::SkVec;
use util::{py_bail, py_pickle_state_impl};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Parameter<T>(SkVec<T>);

impl<T> Parameter<T> {
	fn len(&self) -> usize {
		self.0.len()
	}

	fn check_index(&self, i: usize) -> Result<()> {
		let len = self.len();
		if i >= len {
			let dimension = if len % 10 == 1 && len != 11 {
				"dimension"
			} else {
				"dimensions"
			};
			py_bail!(PyIndexError, "Parameter has {} {}, index {} is out of bounds", self.len(), dimension, i);
		} else {
			Ok(())
		}
	}
}

impl Display for Parameter<f64> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for (i, value) in self.0.iter().enumerate() {
			value.fmt(f)?;
			if i < self.len() - 1 {
				f.write_str(", ")?;
			}
		}
		Ok(())
	}
}

impl Display for Parameter<i64> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for (i, value) in self.0.iter().enumerate() {
			value.fmt(f)?;
			if i < self.len() - 1 {
				f.write_str(", ")?;
			}
		}
		Ok(())
	}
}

impl Display for Parameter<bool> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for (i, value) in self.0.iter().enumerate() {
			if *value {
				f.write_str("True")?;
			} else {
				f.write_str("False")?;
			}
			if i < self.len() - 1 {
				f.write_str(", ")?;
			}
		}
		Ok(())
	}
}

#[derive(Debug)]
#[pyclass(name = "Real", module = "aspartik.b3", sequence, frozen)]
pub struct PyReal {
	inner: Mutex<Parameter<f64>>,
}

#[derive(Debug)]
#[pyclass(name = "Integer", module = "aspartik.b3", sequence, frozen)]
pub struct PyInteger {
	inner: Mutex<Parameter<i64>>,
}

#[derive(Debug)]
#[pyclass(name = "Boolean", module = "aspartik.b3", sequence, frozen)]
pub struct PyBoolean {
	inner: Mutex<Parameter<bool>>,
}

#[rustfmt::skip]
macro_rules! pymethod_impl {
($class:ident, $name:literal, $type:ty, $pytype:literal) => {
	#[pymethods]
	impl $class {
		#[new]
		#[pyo3(signature = (*values))]
		fn new(values: &Bound<PyTuple>) -> Result<Self> {
			check_empty(values)?;

			let values: Vec<$type> = extract(values)?;
		let parameter = Parameter(values.into());
		Ok(Self {
			inner: Mutex::new(parameter),
		})
	}

	fn __len__(&self) -> usize {
		self.inner.lock().len()
	}

	fn __getitem__(&self, i: usize) -> Result<$type> {
		let inner = &*self.inner.lock();
		inner.check_index(i)?;

		Ok(inner.0[i])
	}

	fn __setitem__(
		&self,
		i: usize,
		value: $type,
	) -> Result<()> {
		let inner = &mut *self.inner.lock();
		inner.check_index(i)?;
		inner.0.set(i, value);

		Ok(())
	}

	fn __repr__(&self) -> String {
		let inner = &*self.inner.lock();

		format!("{}({})", $name, inner)
	}

	fn __str__(&self) -> String {
		format!("[{}]", self.inner.lock())
	}

	fn __richcmp__(
		&self,
		other: Bound<PyAny>,
		op: CompareOp,
	) -> Result<bool> {
		let inner = &*self.inner.lock();

		if let Ok(other) = other.downcast::<Self>() {
			let other = &*other.get().inner.lock();

			if inner.0.len() != other.0.len() {
				py_bail!(
					PyValueError,
					"Can't compare parameters of different lengths: {} and {}",
					inner.0.len(), other.0.len()
				);
			}

			return Ok(compare_elements(&inner.0, &other.0, op));
		} else if let Ok(value) = other.extract::<$type>() {
			if matches!(op, CompareOp::Eq | CompareOp::Ne) {
				py_bail!(
					PyTypeError,
					"Can't check equality of {} and {}",
					$name, $pytype,
				)
			}
			return Ok(compare_value(&inner.0, value, op));
		}

		py_bail!(
			PyTypeError,
			"{} can only be compared to other instances or to {}",
			$name, $pytype
		);

	}

	fn accept(&self) {
		self.inner.lock().0.accept();
	}

	fn reject(&self) {
		self.inner.lock().0.reject();
	}

	// pickle
	fn __getnewargs__<'py>(
		&self,
		py: Python<'py>,
	) -> PyResult<Bound<'py, PyTuple>> {
		let inner = &*self.inner.lock();

		PyTuple::new(py, &inner.0)
	}
}
};
}

macro_rules! pymethod_math_impl {
	($class:ident, $name:literal, $type:ty) => {
		#[pymethods]
		impl $class {
			fn __imul__(&self, other: $type) -> Result<()> {
				let inner = &mut *self.inner.lock();
				let p = &mut inner.0;

				for i in 0..p.len() {
					p.set(i, p[i] * other);
				}

				Ok(())
			}

			fn __itruediv__(&self, other: $type) -> Result<()> {
				let inner = &mut *self.inner.lock();
				let p = &mut inner.0;

				for i in 0..p.len() {
					p.set(i, p[i] / other);
				}

				Ok(())
			}
		}
	};
}

pymethod_impl!(PyReal, "Real", f64, "float");
pymethod_impl!(PyInteger, "Integer", i64, "int");
pymethod_impl!(PyBoolean, "Boolean", bool, "bool");

py_pickle_state_impl!(PyReal, _real_pickle_impl);
py_pickle_state_impl!(PyInteger, _integer_pickle_impl);
py_pickle_state_impl!(PyBoolean, _boolean_pickle_impl);

pymethod_math_impl!(PyReal, "Real", f64);
pymethod_math_impl!(PyInteger, "Integer", i64);

#[pymethods]
impl PyReal {
	fn __float__(&self) -> Result<f64> {
		let inner = &*self.inner.lock();
		ensure!(inner.len() == 1, "Tried to coerce a multidimensional parameter to a float");
		Ok(inner.0[0])
	}
}

#[pymethods]
impl PyInteger {
	fn __int__(&self) -> Result<i64> {
		let inner = &*self.inner.lock();
		ensure!(inner.len() == 1, "Tried to coerce a multidimensional parameter to an int");
		Ok(inner.0[0])
	}
}

fn compare_value<T: PartialOrd>(
	values: &SkVec<T>,
	value: T,
	op: CompareOp,
) -> bool {
	values.iter().all(|element| match op {
		CompareOp::Lt => *element < value,
		CompareOp::Le => *element <= value,
		CompareOp::Eq => *element == value,
		CompareOp::Ne => *element != value,
		CompareOp::Gt => *element > value,
		CompareOp::Ge => *element >= value,
	})
}

fn compare_elements<T: PartialOrd>(
	this: &SkVec<T>,
	other: &SkVec<T>,
	op: CompareOp,
) -> bool {
	this.iter().zip(other.iter()).all(|(a, b)| match op {
		CompareOp::Lt => a < b,
		CompareOp::Le => a <= b,
		CompareOp::Eq => a == b,
		CompareOp::Ne => a != b,
		CompareOp::Gt => a > b,
		CompareOp::Ge => a >= b,
	})
}

fn check_empty(values: &Bound<PyTuple>) -> Result<()> {
	if values.is_empty() {
		Err(PyTypeError::new_err(
			"A parameter must have at least one value",
		)
		.into())
	} else {
		Ok(())
	}
}

fn extract<T: for<'a> FromPyObjectBound<'a, 'a>>(
	tuple: &Bound<PyTuple>,
) -> Result<Vec<T>> {
	Ok(tuple.into_iter()
		.map(|v| v.extract::<T>())
		.collect::<PyResult<Vec<T>>>()?)
}
