use anyhow::Result;
use parking_lot::{Mutex, MutexGuard};
use pyo3::prelude::*;
use pyo3::{
	class::basic::CompareOp,
	conversion::FromPyObjectBound,
	exceptions::{PyIndexError, PyTypeError},
	types::PyTuple,
};

use std::{
	fmt::{self, Display},
	sync::Arc,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Parameter {
	Real(Vec<f64>),
	Integer(Vec<i64>),
	Boolean(Vec<bool>),
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

fn compare<T: PartialOrd>(values: &[T], other: T, op: CompareOp) -> bool {
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

#[derive(Debug, Clone)]
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
	inner: Arc<Mutex<Parameter>>,
}

impl PyParameter {
	pub fn inner(&self) -> MutexGuard<Parameter> {
		self.inner.lock()
	}

	pub fn deep_copy(&self) -> PyParameter {
		let inner = &*self.inner();

		Self {
			inner: Arc::new(Mutex::new(inner.clone())),
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
#[expect(non_snake_case)]
impl PyParameter {
	/// Create a new real parameter.
	///
	/// The values will be coerced to a double-precision floating number.
	///
	/// Note that Python will coerce `True` and `False` to 0 and 1, so
	/// `Parameter.Real(True, False)` will succeed a and return a parameter
	/// with values `[0.0, 1.0]`.
	#[staticmethod]
	#[pyo3(signature = (*values))]
	fn Real(values: &Bound<PyTuple>) -> Result<Self> {
		check_empty(values)?;

		let values: Vec<f64> = extract(values)?;
		let parameter = Parameter::Real(values);
		Ok(Self {
			inner: Arc::new(Mutex::new(parameter)),
		})
	}

	/// Create a new integer parameter.
	///
	/// Note that Python will coerce `True` and `False` to 0 and 1, so
	/// `Parameter.Integer(True, False)` will succeed a and return a
	/// parameter with values `[0, 1]`.
	#[staticmethod]
	#[pyo3(signature = (*values))]
	fn Integer(values: &Bound<PyTuple>) -> Result<Self> {
		check_empty(values)?;

		let values: Vec<i64> = extract(values)?;
		let parameter = Parameter::Integer(values);
		Ok(Self {
			inner: Arc::new(Mutex::new(parameter)),
		})
	}

	/// Create a new boolean parameter.
	#[staticmethod]
	#[pyo3(signature = (*values))]
	fn Boolean(values: &Bound<PyTuple>) -> Result<Self> {
		check_empty(values)?;

		let values: Vec<bool> = extract(values)?;
		let parameter = Parameter::Boolean(values);
		Ok(Self {
			inner: Arc::new(Mutex::new(parameter)),
		})
	}

	fn __len__(&self) -> Result<usize> {
		Ok(self.inner().len())
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
				p[i] = value;
			}
			Parameter::Integer(p) => {
				let value = value.extract::<i64>()?;
				p[i] = value;
			}
			Parameter::Boolean(p) => {
				let value = value.extract::<bool>()?;
				p[i] = value;
			}
		}

		Ok(())
	}

	fn __repr__(&self) -> Result<String> {
		let inner = &*self.inner();

		let subtype = match inner {
			Parameter::Real(..) => "Real",
			Parameter::Integer(..) => "Integer",
			Parameter::Boolean(..) => "Boolean",
		};

		Ok(format!("Parameter.{}({})", subtype, inner))
	}

	fn __str__(&self) -> Result<String> {
		Ok(format!("[{}]", self.inner()))
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

	fn is_real(&self) -> Result<bool> {
		Ok(matches!(&*self.inner(), Parameter::Real(_)))
	}

	fn is_integer(&self) -> Result<bool> {
		Ok(matches!(&*self.inner(), Parameter::Integer(_)))
	}

	fn is_boolean(&self) -> Result<bool> {
		Ok(matches!(&*self.inner(), Parameter::Boolean(_)))
	}
}

fn extract<T: for<'a> FromPyObjectBound<'a, 'a>>(
	tuple: &Bound<PyTuple>,
) -> Result<Vec<T>> {
	Ok(tuple.into_iter()
		.map(|v| v.extract::<T>())
		.collect::<PyResult<Vec<T>>>()?)
}
