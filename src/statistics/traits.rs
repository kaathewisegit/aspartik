/// The `Min` trait specifies than an object has a minimum value
pub trait Min<T> {
	/// Returns the minimum value in the domain of a given distribution
	/// if it exists, otherwise `None`.
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Min;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.0, n.min());
	/// ```
	fn min(&self) -> T;
}

/// The `Max` trait specifies that an object has a maximum value
pub trait Max<T> {
	/// Returns the maximum value in the domain of a given distribution
	/// if it exists, otherwise `None`.
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Max;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(1.0, n.max());
	/// ```
	fn max(&self) -> T;
}

pub trait Distribution {
	/// Mean, or `None` if it doesn't exist
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Distribution;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.5, n.mean().unwrap());
	/// ```
	fn mean(&self) -> Option<f64> {
		None
	}

	/// Variance, or `None` if it doesn't exist
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Distribution;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(1.0 / 12.0, n.variance().unwrap());
	/// ```
	fn variance(&self) -> Option<f64> {
		None
	}

	/// Standard deviation, or `None` if it doesn't exist
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Distribution;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!((1f64 / 12f64).sqrt(), n.std_dev().unwrap());
	/// ```
	fn std_dev(&self) -> Option<f64> {
		self.variance().map(|var| var.sqrt())
	}

	/// Entropy, or `None` if it doesn't exist
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Distribution;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.0, n.entropy().unwrap());
	/// ```
	fn entropy(&self) -> Option<f64> {
		None
	}

	/// Skewness, of `None` if it doesn't exist
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Distribution;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.0, n.skewness().unwrap());
	/// ```
	fn skewness(&self) -> Option<f64> {
		None
	}
}

/// The `Mean` trait implements the calculation of a mean.
// TODO: Clarify the traits of multidimensional distributions
pub trait MeanN<T> {
	fn mean(&self) -> Option<T>;
}

// TODO: Clarify the traits of multidimensional distributions
pub trait VarianceN<T> {
	fn variance(&self) -> Option<T>;
}

/// The `Median` trait returns the median of the distribution.
pub trait Median {
	/// Returns the median.
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Median;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(0.5, n.median());
	/// ```
	fn median(&self) -> f64;
}

/// The `Mode` trait specifies that an object has a closed form solution
/// for its mode(s)
pub trait Mode<T> {
	/// Returns the mode, if one exists.
	///
	/// # Examples
	///
	/// ```
	/// use statrs::statistics::Mode;
	/// use statrs::distribution::Uniform;
	///
	/// let n = Uniform::new(0.0, 1.0).unwrap();
	/// assert_eq!(Some(0.5), n.mode());
	/// ```
	fn mode(&self) -> T;
}
