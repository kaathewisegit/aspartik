macro_rules! impl_pymethods {
	(for $class:ty;) => {};
	(
		for $class:ty;
		new($($arg:ident: $type:ty),* $(,)?) throws $err:ty;
		$(get($field:ident: $get_type:ty as $py_field_name:ident);)*
		repr($repr_fmt:literal, $($repr_args:tt),* $(,)?);
		$(Continuous $continuous:literal;)?
		$(ContinuousCDF $continuous_cdf:literal;)?
		$(Discrete $discrete:literal;)?
		$(DiscreteCDF $discrete_cdf:literal;)?
		$(Distribution $distribution:literal;)?

		$(@ $($rest:tt)*)?
	) => {
		#[cfg(feature = "python")]
		use rng::PyRng;

		#[cfg(feature = "python")]
		#[pymethods]
		impl $class {
			#[new]
			fn py_new($($arg: $type),*) -> Result<$class, $err> {
				<$class>::new($($arg),*)
			}

			$(
			#[getter($field)]
			fn $py_field_name(&self) -> $get_type {
				self.$field
			}
			)*

			fn __repr__(&self) -> String {
				format!($repr_fmt, $(self.$repr_args),*)
			}

			$(
			#[pyo3(name = "pdf")]
			fn py_pdf(&self, x: f64) -> f64 {
				#[expect(clippy::no_effect)]
				$continuous;
				self.pdf(x)
			}

			#[pyo3(name = "ln_pdf")]
			fn py_ln_pdf(&self, x: f64) -> f64 {
				self.ln_pdf(x)
			}
			)?

			$(
			#[pyo3(name = "cdf")]
			fn py_cdf(&self, x: f64) -> f64 {
				#[expect(clippy::no_effect)]
				$continuous_cdf;
				self.cdf(x)
			}

			#[pyo3(name = "sf")]
			fn py_sf(&self, x: f64) -> f64 {
				self.sf(x)
			}

			#[pyo3(name = "inverse_cdf")]
			fn py_inverse_cdf(
				&self, p: math::Probability<f64>
			) -> f64 {
				self.inverse_cdf(p)
			}

			#[getter]
			#[pyo3(name = "lower")]
			fn py_lower(&self) -> f64 {
				self.lower()
			}

			#[getter]
			#[pyo3(name = "upper")]
			fn py_upper(&self) -> f64 {
				self.upper()
			}
			)?

			$(
			#[pyo3(name = "pmf")]
			fn py_pmf(&self, x: <Self as Discrete>::T) -> f64 {
				#[expect(clippy::no_effect)]
				$discrete;
				self.pmf(x)
			}

			#[pyo3(name = "ln_pmf")]
			fn py_ln_pmf(&self, x: <Self as Discrete>::T) -> f64 {
				self.ln_pmf(x)
			}
			)?

			$(
			#[pyo3(name = "cdf")]
			fn py_cdf(&self, x: <Self as Discrete>::T) -> f64 {
				#[expect(clippy::no_effect)]
				$discrete_cdf;
				self.cdf(x)
			}

			#[pyo3(name = "sf")]
			fn py_sf(&self, x: <Self as Discrete>::T) -> f64 {
				self.sf(x)
			}

			#[pyo3(name = "inverse_cdf")]
			fn py_inverse_cdf(
				&self, p: math::Probability<f64>,
			) -> <Self as Discrete>::T {
				self.inverse_cdf(p)
			}

			#[getter]
			#[pyo3(name = "lower")]
			fn py_lower(&self) -> <Self as Discrete>::T {
				self.lower()
			}

			#[getter]
			#[pyo3(name = "upper")]
			fn py_upper(&self) -> <Self as Discrete>::T {
				self.upper()
			}
			)?

			$(
			#[pyo3(name = "mean")]
			fn py_mean(&self) -> Option<f64> {
				#[expect(clippy::no_effect)]
				$distribution;
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
			)?

			#[pyo3(name = "sample")]
			fn py_sample(&self, rng: Py<PyRng>) -> PyResult<f64> {
				use rand::distr::Distribution;

				let x = self.sample(&mut rng.get().inner());
				Ok(x)
			}

			$($($rest)*)?
		}
	};
}
pub(crate) use impl_pymethods;
