macro_rules! impl_pyerr {
	($err: ty, $pyexc: ty) => {
		impl std::convert::From<$err> for PyErr {
			fn from(err: $err) -> PyErr {
				<$pyexc>::new_err(err)
			}
		}
	};
}
pub(crate) use impl_pyerr;

macro_rules! impl_pymethods {
	(for $class:ty;) => {};
	(
		for $class:ty;
		new($($arg:ident: $type:ty),* $(,)?) throws $err:ty;
		$($rest:tt)*
	) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[new]
			fn py_new($($arg: $type),*) -> Result<$class, $err> {
				<$class>::new($($arg),*)
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; get($m:ident) $field:ident: $type:ty; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[getter($field)]
			fn $m(&self) -> $type {
				self.$field
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; get(as $m:ident) $field:ident: $type:ty; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[getter]
			fn $m(&self) -> $type {
				self.$field
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(
		for $class:ty;
		repr($fmt:literal, $($args:tt),* $(,)?);
		$($rest:tt)*
	) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			fn __repr__(&self) -> String {
				format!($fmt, $(self.$args),*)
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; Continuous; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[pyo3(name = "pdf")]
			fn py_pdf(&self, x: <Self as Continuous>::T) -> f64 {
				self.pdf(x)
			}

			#[pyo3(name = "ln_pdf")]
			fn py_ln_pdf(&self, x: <Self as Continuous>::T) -> f64 {
				self.ln_pdf(x)
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; ContinuousCDF; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[pyo3(name = "cdf")]
			fn py_cdf(&self, x: <Self as Continuous>::T) -> f64 {
				self.cdf(x)
			}

			#[pyo3(name = "sf")]
			fn py_sf(&self, x: <Self as Continuous>::T) -> f64 {
				self.sf(x)
			}

			#[pyo3(name = "inverse_cdf")]
			fn py_inverse_cdf(&self, p: f64) -> <Self as Continuous>::T {
				self.inverse_cdf(p)
			}

			#[pyo3(name = "lower")]
			fn py_lower(&self) -> <Self as Continuous>::T {
				self.lower()
			}

			#[pyo3(name = "upper")]
			fn py_upper(&self) -> <Self as Continuous>::T {
				self.upper()
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; Discrete; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[pyo3(name = "pmf")]
			fn py_pmf(&self, x: <Self as Discrete>::T) -> f64 {
				self.pmf(x)
			}

			#[pyo3(name = "ln_pmf")]
			fn py_ln_pmf(&self, x: <Self as Discrete>::T) -> f64 {
				self.ln_pmf(x)
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; DiscreteCDF; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[pyo3(name = "cdf")]
			fn py_cdf(&self, x: <Self as Discrete>::T) -> f64 {
				self.cdf(x)
			}

			#[pyo3(name = "sf")]
			fn py_sf(&self, x: <Self as Discrete>::T) -> f64 {
				self.sf(x)
			}

			#[pyo3(name = "inverse_cdf")]
			fn py_inverse_cdf(&self, p: f64) -> <Self as Discrete>::T {
				self.inverse_cdf(p)
			}

			#[pyo3(name = "lower")]
			fn py_lower(&self) -> <Self as Discrete>::T {
				self.lower()
			}

			#[pyo3(name = "upper")]
			fn py_upper(&self) -> <Self as Discrete>::T {
				self.upper()
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; Distribution; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[pyo3(name = "mean")]
			fn py_mean(&self) -> Option<f64> {
				self.mean()
			}

			#[pyo3(name = "median")]
			fn py_median(&self) -> Option<f64> {
				self.median()
			}

			#[pyo3(name = "variance")]
			fn py_variance(&self) -> Option<f64> {
				self.variance()
			}

			#[pyo3(name = "std_dev")]
			fn py_std_dev(&self) -> Option<f64> {
				self.std_dev()
			}

			#[pyo3(name = "entropy")]
			fn py_entropy(&self) -> Option<f64> {
				self.entropy()
			}

			#[pyo3(name = "skewness")]
			fn py_skewness(&self) -> Option<f64> {
				self.skewness()
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
	(for $class:ty; sample; $($rest:tt)*) => {
		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[pyo3(name = "sample")]
			fn py_sample(&self, rng: Bound<'_, PyAny>) -> f64 {
				use rng::PyRng;
				use rand::distr::Distribution;

				let rng = PyRng::downcast(rng);

				let x = self.sample(&mut rng.get().inner());
				x
			}
		}

		impl_pymethods!(for $class; $($rest)*);
	};
}
pub(crate) use impl_pymethods;
