use anyhow::{bail, ensure, Result};
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use pyo3::{
	class::basic::CompareOp,
	conversion::FromPyObjectBound,
	exceptions::{PyIndexError, PyTypeError},
	types::{PyTuple, PyType},
};

use std::fmt::{self, Display};

use skvec::SkVec;

#[derive(Debug, Clone, PartialEq)]
pub enum Parameter {
	Real(SkVec<f64>),
	Integer(SkVec<i64>),
	Boolean(SkVec<bool>),
}

impl Parameter {
	fn len(&self) -> usize {
		match self {
			Parameter::Real(p) => p.len(),
			Parameter::Integer(p) => p.len(),
			Parameter::Boolean(p) => p.len(),
		}
	}

	fn check_index(&self, i: usize) -> Result<()> {
		if i >= self.len() {
			let dimension = if self.len() % 10 == 1 {
				"dimension"
			} else {
				"dimensions"
			};
			Err(PyIndexError::new_err(
				format!("Parameter has {} {}, index {} is out of bounds", self.len(), dimension, i)
			).into())
		} else {
			Ok(())
		}
	}
}

fn compare<T: PartialOrd>(values: &SkVec<T>, other: T, op: CompareOp) -> bool {
	values.iter().all(|v| match op {
		CompareOp::Lt => *v < other,
		CompareOp::Le => *v <= other,
		CompareOp::Eq => *v == other,
		CompareOp::Ne => *v != other,
		CompareOp::Gt => *v > other,
		CompareOp::Ge => *v >= other,
	})
}

impl Display for Parameter {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Parameter::Real(p) => {
				for (i, value) in p.iter().enumerate() {
					value.fmt(f)?;
					if i < p.len() - 1 {
						f.write_str(", ")?;
					}
				}
			}
			Parameter::Integer(p) => {
				for (i, value) in p.iter().enumerate() {
					value.fmt(f)?;
					if i < p.len() - 1 {
						f.write_str(", ")?;
					}
				}
			}
			Parameter::Boolean(p) => {
				for (i, value) in p.iter().enumerate() {
					if *value {
						f.write_str("True")?;
					} else {
						f.write_str("False")?;
					}
					if i < p.len() - 1 {
						f.write_str(", ")?;
					}
				}
			}
		}

		Ok(())
	}
}

#[derive(Debug)]
#[pyclass(name = "Parameter", module = "aspartik.b3", sequence, frozen)]
/// Represents dimensional parameters which can hold arbitrary numbers.
///
/// This class has no constructor.  Instead, it's static methods `Real`,
/// `Integer`, and `Boolean` can be used to create parameters made up of
/// doubles, 64-bit signed integers, or booleans respectively.  A parameter can
/// only hold one type: it cannot mix integers and floats, for example.  It also
/// cannot change the number of dimensions after creation.
///
/// The parameter values can be accessed using indexing.  Dimensions are
/// zero-indexed, so `param[0]` is the first value, `param[1]` is the second,
/// and so on.
pub struct PyParameter {
	inner: Mutex<Parameter>,
}

impl PyParameter {
	pub fn inner(&self) -> MutexGuard<Parameter> {
		self.inner.lock()
	}

	pub fn accept(&self) {
		let inner = &mut *self.inner();

		match inner {
			Parameter::Real(p) => p.accept(),
			Parameter::Integer(p) => p.accept(),
			Parameter::Boolean(p) => p.accept(),
		}
	}

	pub fn reject(&self) {
		let inner = &mut *self.inner();

		match inner {
			Parameter::Real(p) => p.reject(),
			Parameter::Integer(p) => p.reject(),
			Parameter::Boolean(p) => p.reject(),
		}
	}
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

#[pymethods]
impl PyParameter {
	/// Create a new real parameter.
	///
	/// The values will be coerced to a double-precision floating number.
	///
	/// Note that Python will coerce `True` and `False` to 0 and 1, so
	/// `Parameter.Real(True, False)` will succeed a and return a parameter
	/// with values `[0.0, 1.0]`.
	#[classmethod]
	#[pyo3(name = "Real", signature = (*values))]
	fn real(_cls: Py<PyType>, values: &Bound<PyTuple>) -> Result<Self> {
		check_empty(values)?;

		let values: Vec<f64> = extract(values)?;
		let parameter = Parameter::Real(values.into());
		Ok(Self {
			inner: Mutex::new(parameter),
		})
	}

	/// Create a new integer parameter.
	///
	/// Note that Python will coerce `True` and `False` to 0 and 1, so
	/// `Parameter.Integer(True, False)` will succeed a and return a
	/// parameter with values `[0, 1]`.
	#[classmethod]
	#[pyo3(name = "Integer", signature = (*values))]
	fn integer(_cls: Py<PyType>, values: &Bound<PyTuple>) -> Result<Self> {
		check_empty(values)?;

		let values: Vec<i64> = extract(values)?;
		let parameter = Parameter::Integer(values.into());
		Ok(Self {
			inner: Mutex::new(parameter),
		})
	}

	/// Create a new boolean parameter.
	#[classmethod]
	#[pyo3(name = "Boolean", signature = (*values))]
	fn boolean(_cls: Py<PyType>, values: &Bound<PyTuple>) -> Result<Self> {
		check_empty(values)?;

		let values: Vec<bool> = extract(values)?;
		let parameter = Parameter::Boolean(values.into());
		Ok(Self {
			inner: Mutex::new(parameter),
		})
	}

	fn __len__(&self) -> usize {
		self.inner().len()
	}

	fn __getitem__(&self, py: Python, i: usize) -> Result<PyObject> {
		let inner = &*self.inner();
		inner.check_index(i)?;

		Ok(match inner {
			Parameter::Real(p) => p[i].into_pyobject(py)?.into(),
			Parameter::Integer(p) => p[i].into_pyobject(py)?.into(),
			Parameter::Boolean(p) => {
				p[i].into_pyobject(py)?.to_owned().into()
			}
		})
	}

	fn __setitem__(&self, i: usize, value: Bound<PyAny>) -> Result<()> {
		let inner = &mut *self.inner();
		inner.check_index(i)?;

		match inner {
			Parameter::Real(p) => {
				let value = value.extract::<f64>()?;
				p.set(i, value);
			}
			Parameter::Integer(p) => {
				let value = value.extract::<i64>()?;
				p.set(i, value);
			}
			Parameter::Boolean(p) => {
				let value = value.extract::<bool>()?;
				p.set(i, value);
			}
		}

		Ok(())
	}

	fn __repr__(&self) -> String {
		let inner = &*self.inner();

		let subtype = match inner {
			Parameter::Real(..) => "Real",
			Parameter::Integer(..) => "Integer",
			Parameter::Boolean(..) => "Boolean",
		};

		format!("Parameter.{}({})", subtype, inner)
	}

	fn __str__(&self) -> String {
		format!("[{}]", self.inner())
	}

	fn __richcmp__(
		&self,
		other: Bound<PyAny>,
		op: CompareOp,
	) -> Result<bool> {
		let inner = &*self.inner();

		match inner {
			Parameter::Real(p) => {
				let other = other.extract::<f64>()?;
				Ok(compare(p, other, op))
			}
			Parameter::Integer(p) => {
				let other = other.extract::<i64>()?;
				Ok(compare(p, other, op))
			}
			Parameter::Boolean(p) => {
				let other = other.extract::<bool>()?;
				Ok(compare(p, other, op))
			}
		}
	}

	fn __imul__(&self, other: Bound<PyAny>) -> Result<()> {
		let inner = &mut *self.inner();

		match inner {
			Parameter::Real(ref mut p) => {
				let other = other.extract::<f64>()?;
				for i in 0..p.len() {
					p.set(i, p[i] * other);
				}
			}
			Parameter::Integer(p) => {
				let other = other.extract::<i64>()?;
				for i in 0..p.len() {
					p.set(i, p[i] * other);
				}
			}
			Parameter::Boolean(_) => {
				bail!("Booleans don't support multiplication");
			}
		}

		Ok(())
	}

	fn __itruediv__(&self, other: Bound<PyAny>) -> Result<()> {
		let inner = &mut *self.inner();

		match inner {
			Parameter::Real(ref mut p) => {
				let other = other.extract::<f64>()?;
				for i in 0..p.len() {
					p.set(i, p[i] / other);
				}
			}
			Parameter::Integer(p) => {
				let other = other.extract::<i64>()?;
				for i in 0..p.len() {
					p.set(i, p[i] / other);
				}
			}
			Parameter::Boolean(_) => {
				bail!("Booleans don't support divison");
			}
		}

		Ok(())
	}

	pub fn is_real(&self) -> bool {
		matches!(&*self.inner(), Parameter::Real(..))
	}

	pub fn is_integer(&self) -> bool {
		matches!(&*self.inner(), Parameter::Integer(..))
	}

	pub fn is_boolean(&self) -> bool {
		matches!(&*self.inner(), Parameter::Boolean(..))
	}

	pub fn one_real(&self) -> Result<f64> {
		let inner = &*self.inner();
		ensure!(
			inner.len() == 1,
			"The parameter is not one-dimensional"
		);

		match inner {
			Parameter::Real(p) => Ok(p[0]),
			_ => bail!("Not a real parameter"),
		}
	}
}

fn extract<T: for<'a> FromPyObjectBound<'a, 'a>>(
	tuple: &Bound<PyTuple>,
) -> Result<Vec<T>> {
	Ok(tuple.into_iter()
		.map(|v| v.extract::<T>())
		.collect::<PyResult<Vec<T>>>()?)
}
