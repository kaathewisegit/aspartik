use anyhow::Result;
use computare_distributions::{Continuous, Statistics};
use pyo3::{exceptions::PyTypeError, prelude::*};

use rng::PyRng;
use util::{py_bail, py_call_method, py_has_method};

macro_rules! wrapper {
	(
		$class:ident {
			$($field:ident),*
		};
		repr($repr_fmt:literal, $($repr_args:tt),* $(,)?);
		$(Continuous $continuous:literal;)?
		$(Discrete $discrete:literal;)?
		$(Statistics $statistics:literal;)?

		$(@ $($rest:tt)*)?
	) => {
		#[derive(Debug)]
		#[pyclass(module = "aspartik.distributions", frozen)]
		struct $class {
			$(#[pyo3(get)] $field: Py<PyAny>),*
		}

		impl $class {
			fn make(&self, py: Python) -> PyResult<computare_distributions::$class> {
				Ok(computare_distributions::$class::new($(self.$field.extract::<f64>(py)?),*))
			}
		}

		#[pymethods]
		impl $class {
			#[new]
			fn new(py: Python, $($field: Py<PyAny>),*) -> PyResult<Self> {
				$({
				let $field = $field.bind(py);
				if !py_has_method!($field, "__float__") {
					py_bail!(PyTypeError, "{} must implement the `SupportsFloat` protocol", $field);
				}
				})*
				Ok(Self { $($field),* })
			}

			fn __repr__(&self) -> String {
				format!($repr_fmt, $(self.$repr_args),*)
			}

			$(
			fn pdf(&self, py: Python, x: f64) -> PyResult<f64> {
				#[expect(clippy::no_effect)]
				$continuous;
				Ok(self.make(py)?.pdf(x))
			}

			fn ln_pdf(&self, py: Python, x: f64) -> PyResult<f64> {
				Ok(self.make(py)?.ln_pdf(x))
			}

			fn cdf(&self, py: Python, x: f64) -> PyResult<f64> {
				Ok(self.make(py)?.cdf(x))
			}

			fn inverse_cdf(&self, py: Python, p: f64) -> PyResult<f64> {
				Ok(self.make(py)?.inverse_cdf(p))
			}

			#[getter]
			fn lower(&self, py: Python) -> PyResult<f64> {
				Ok(self.make(py)?.lower())
			}

			#[getter]
			#[pyo3(name = "upper")]
			fn upper(&self, py: Python) -> PyResult<f64> {
				Ok(self.make(py)?.upper())
			}
			)?

			$(
			fn mean(&self, py: Python) -> Result<f64> {
				#[expect(clippy::no_effect)]
				$statistics;
				Ok(self.make(py)?.mean()?)
			}

			fn median(&self, py: Python) -> Result<f64> {
				Ok(self.make(py)?.median()?)
			}

			fn mode(&self, py: Python) -> Result<f64> {
				Ok(self.make(py)?.mode()?)
			}

			fn variance(&self, py: Python) -> Result<f64> {
				Ok(self.make(py)?.variance()?)
			}

			fn std(&self, py: Python) -> Result<f64> {
				Ok(self.make(py)?.std()?)
			}

			fn entropy(&self, py: Python) -> Result<f64> {
				Ok(self.make(py)?.entropy()?)
			}
			)?

			fn sample(&self, py: Python, rng: Py<PyRng>) -> PyResult<f64> {
				use rand::distr::Distribution;

				let x = self.make(py)?.sample(&mut rng.get().inner());
				Ok(x)
			}

			fn is_changed(&self, py: Python) -> PyResult<bool> {
				$(
				let $field = self.$field.bind(py);
				if py_has_method!($field, "is_changed") && py_call_method!($field, "is_changed")?.extract::<bool>()? {
					return Ok(true);
				}
				)*

				Ok(false)
			}

			$($($rest)*)?
		}
	};
}

// TODO: Beta, Gamma, InverseGamma, Normal, and Uniform have `&str` error types, which can't be
// wrapped by anyhow
wrapper! {
	Beta { shape_a, shape_b };
	repr("Beta(shape_a={}, shape_b={})", shape_a, shape_b);
	Continuous true;
}

wrapper! {
	Exponential { rate };
	repr("Exponential(rate={})", rate);
	Continuous true;
	Statistics true;
}

wrapper! {
	Gamma { shape, scale };
	repr("InverseGamma(shape={}, scale={})", shape, scale);
	Continuous true;
}

wrapper! {
	InverseGamma { shape, scale };
	repr("InverseGamma(shape={}, scale={})", shape, scale);
	Continuous true;
}

wrapper! {
	Laplace { location, scale };
	repr("Laplace(location={}, scale={})", location, scale);
	Continuous true;
	Statistics true;
}

wrapper! {
	LogNormal { location, scale };
	repr("LogNormal(location={}, scale={})", location, scale);
	Continuous true;
	Statistics true;
}

wrapper! {
	Normal { mean, std };
	repr("Normal(mean={}, std={})", mean, std);
	Continuous true;
}

wrapper! {
	Uniform { min, max };
	repr("Uniform(min={}, max={})", min, max);
	Continuous true;
}

#[pymodule(name = "_distributions_rust_impl")]
pub mod pymodule {
	use super::*;

	#[pymodule_export]
	use super::{
		Beta, Exponential, Gamma, InverseGamma, Laplace, LogNormal,
		Normal, Uniform,
	};

	#[pymodule_init]
	fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
		::util::py_patch_module!(m);

		Ok(())
	}
}
