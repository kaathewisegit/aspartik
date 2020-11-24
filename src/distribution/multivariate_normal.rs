use crate::distribution::Continuous;
use crate::distribution::Normal;
use crate::statistics::{Covariance, Max, MeanN, Min, Mode};
use crate::{Result, StatsError};
use nalgebra::{
    base::allocator::Allocator,
    base::{dimension::DimName, MatrixN, VectorN},
    Cholesky, DefaultAllocator, Dim, DimMin, LU, U1,
};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;
use std::f64::consts::{E, PI};

/// Implements the [Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
/// distribution using the "nalgebra" crate for matrix operations
///
/// # Examples
///
/// ```
/// use statrs::distribution::{MultivariateNormal, Continuous};
/// use nalgebra::{Vector2, Matrix2};
/// use statrs::statistics::{MeanN, Covariance};
///
/// let mvn = MultivariateNormal::new(Vector2::zeros(), Matrix2::identity()).unwrap();
/// assert_eq!(mvn.mean(), Vector2::new(0.,  0.));
/// assert_eq!(mvn.variance(), Matrix2::new(1., 0., 0., 1.));
/// assert_eq!(mvn.pdf(Vector2::new(1.,  1.)), 0.05854983152431917);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    dim: usize,
    cov_chol_decomp: MatrixN<f64, N>,
    mu: VectorN<f64, N>,
    cov: MatrixN<f64, N>,
    precision: MatrixN<f64, N>,
    pdf_const: f64,
}

impl<N> MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    ///  Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix `cov`
    ///
    /// # Errors
    ///
    /// Returns an error if the given covariance matrix is not
    /// symmetric or positive-definite
    pub fn new(mean: VectorN<f64, N>, cov: MatrixN<f64, N>) -> Result<Self> {
        let dim = mean.len();
        // Check that the provided covariance matrix is symmetric
        // Check that mean and covariance do not contain NaN
        if cov.lower_triangle() != cov.upper_triangle().transpose()
            || mean.iter().any(|f| f.is_nan())
            || cov.iter().any(|f| f.is_nan())
        {
            return Err(StatsError::BadParams);
        }
        let cov_det = LU::new(cov.clone()).determinant();
        let pdf_const = ((2. * PI).powi(mean.nrows() as i32) * cov_det.abs())
            .recip()
            .sqrt();
        // Store the Cholesky decomposition of the covariance matrix
        // for sampling
        match Cholesky::new(cov.clone()) {
            None => Err(StatsError::BadParams),
            Some(cholesky_decomp) => {
                let precision = cholesky_decomp.inverse();
                Ok(MultivariateNormal {
                    dim,
                    cov_chol_decomp: cholesky_decomp.unpack(),
                    mu: mean,
                    cov,
                    precision,
                    pdf_const,
                })
            }
        }
    }
    /// Returns the entropy of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * ln(det(2 * π * e * Σ))
    /// ```
    ///
    /// where `Σ` is the covariance matrix and `det` is the determinant
    pub fn entropy(&self) -> Option<f64> {
        Some(
            0.5 * LU::new(self.variance().scale(2. * PI * E))
                .determinant()
                .ln(),
        )
    }
}

impl<N> Distribution<VectorN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Samples from the multivariate normal distribution
    ///
    /// # Formula
    /// L * Z + μ
    ///
    /// where `L` is the Cholesky decomposition of the covariance matrix,
    /// `Z` is a vector of normally distributed random variables, and
    /// `μ` is the mean vector

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> VectorN<f64, N> {
        let d = Normal::new(0., 1.).unwrap();
        let z = VectorN::<f64, N>::from_distribution(&d, rng);
        (&self.cov_chol_decomp * z) + &self.mu
    }
}

impl<N> Min<Vec<f64>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn min(&self) -> Vec<f64> {
        vec![f64::NEG_INFINITY; self.dim]
    }
}

impl<N> Max<Vec<f64>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the maximum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn max(&self) -> Vec<f64> {
        vec![f64::INFINITY; self.dim]
    }
}

impl<N> MeanN<N> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the mean of the normal distribution
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution
    fn mean(&self) -> VectorN<f64, N> {
        self.mu.clone()
    }
}

impl<N> Covariance<N> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the covariance matrix of the multivariate normal distribution
    fn variance(&self) -> MatrixN<f64, N> {
        self.cov.clone()
    }
}

impl<N> Mode<VectorN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the mode of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the mean
    fn mode(&self) -> VectorN<f64, N> {
        self.mu.clone()
    }
}

impl<N> Continuous<VectorN<f64, N>, f64> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Calculates the probability density function for the multivariate
    /// normal distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2 * π) ^ (-k / 2) * det(Σ) ^ (1 / 2) * e ^ ( -(1 / 2) * transpose(x - μ) * inv(Σ) * (x - μ))
    /// ```
    ///
    /// where `μ` is the mean, `inv(Σ)` is the precision matrix, `det(Σ)` is the determinant
    /// of the covariance matrix, and `k` is the dimension of the distribution
    fn pdf(&self, x: VectorN<f64, N>) -> f64 {
        let dv = x - &self.mu;
        let exp_term = -0.5
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const * exp_term.exp()
    }
    /// Calculates the log probability density function for the multivariate
    /// normal distribution at `x`. Equivalent to pdf(x).ln().
    fn ln_pdf(&self, x: VectorN<f64, N>) -> f64 {
        let dv = x - &self.mu;
        let exp_term = -0.5
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const.ln() + exp_term
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests  {
    use crate::distribution::{Continuous, MultivariateNormal};
    use crate::statistics::*;
    use crate::consts::ACC;
    use core::fmt::Debug;
    use nalgebra::base::allocator::Allocator;
    use nalgebra::{
        DefaultAllocator, Dim, DimMin, DimName, Matrix2, Matrix3, MatrixN, Vector2, Vector3,
        VectorN, U1, U2,
    };

    fn try_create<N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>) -> MultivariateNormal<N>
    where
        N: Dim + DimMin<N, Output = N> + DimName,
        DefaultAllocator: Allocator<f64, N>,
        DefaultAllocator: Allocator<f64, N, N>,
        DefaultAllocator: Allocator<f64, U1, N>,
        DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = MultivariateNormal::new(mean, covariance);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn create_case<N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>)
    where
        N: Dim + DimMin<N, Output = N> + DimName,
        DefaultAllocator: Allocator<f64, N>,
        DefaultAllocator: Allocator<f64, N, N>,
        DefaultAllocator: Allocator<f64, U1, N>,
        DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = try_create(mean.clone(), covariance.clone());
        assert_eq!(mean, mvn.mean());
        assert_eq!(covariance, mvn.variance());
    }

    fn bad_create_case<N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>)
    where
        N: Dim + DimMin<N, Output = N> + DimName,
        DefaultAllocator: Allocator<f64, N>,
        DefaultAllocator: Allocator<f64, N, N>,
        DefaultAllocator: Allocator<f64, U1, N>,
        DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = MultivariateNormal::new(mean, covariance);
        assert!(mvn.is_err());
    }

    fn test_case<T, F, N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>, expected: T, eval: F)
    where
        T: Debug + PartialEq,
        F: Fn(MultivariateNormal<N>) -> T,
        N: Dim + DimMin<N, Output = N> + DimName,
        DefaultAllocator: Allocator<f64, N>,
        DefaultAllocator: Allocator<f64, N, N>,
        DefaultAllocator: Allocator<f64, U1, N>,
        DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_eq!(expected, x);
    }

    fn test_almost<F, N>(
        mean: VectorN<f64, N>,
        covariance: MatrixN<f64, N>,
        expected: f64,
        acc: f64,
        eval: F,
    ) where
        F: Fn(MultivariateNormal<N>) -> f64,
        N: Dim + DimMin<N, Output = N> + DimName,
        DefaultAllocator: Allocator<f64, N>,
        DefaultAllocator: Allocator<f64, N, N>,
        DefaultAllocator: Allocator<f64, U1, N>,
        DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_almost_eq!(expected, x, acc);
    }

    macro_rules! vect2 {
        ($x11:expr, $x12:expr) => (Vector2::new($x11, $x12));
    }

    macro_rules! vect3 {
        ($x11:expr, $x12:expr, $x13:expr) => (Vector3::new($x11, $x12, $x13));
    }

    macro_rules! mat2 {
        ($x11:expr, $x12:expr, $x21:expr, $x22:expr) => (Matrix2::new($x11, $x12, $x21, $x22));
    }

    macro_rules! mat3 {
        ($x11:expr, $x12:expr, $x13:expr, $x21:expr, $x22:expr, $x23:expr, $x31:expr, $x32:expr, $x33:expr) => (Matrix3::new($x11, $x12, $x13, $x21, $x22, $x23, $x31, $x32, $x33));
    }

    #[test]
    fn test_create() {
        create_case(vect2![0., 0.], mat2![1., 0., 0., 1.]);
        create_case(vect2![10.,  5.], mat2![2., 1., 1., 2.]);
        create_case(vect3![4., 5., 6.], mat3![2., 1., 0., 1., 2., 1., 0., 1., 2.]);
        create_case(vect2![0., f64::INFINITY], Matrix2::identity());
        create_case(Vector2::zeros(), mat2![f64::INFINITY, 0., 0., f64::INFINITY]);
    }

    #[test]
    fn test_bad_create() {
        // Covariance not symmetric
        bad_create_case(Vector2::zeros(), mat2![1., 1., 0., 1.]);
        // Covariance not positive-definite
        bad_create_case(Vector2::zeros(), mat2![1., 2., 2., 1.]);
        // NaN in mean
        bad_create_case(vect2![0., f64::NAN], Matrix2::identity());
        // NaN in Covariance Matrix
        bad_create_case(Vector2::zeros(), mat2![1., 0., 0., f64::NAN]);
    }

    #[test]
    fn test_variance() {
        let variance = |x: MultivariateNormal<U2>| x.variance();
        test_case(Vector2::zeros(), Matrix2::identity(), mat2![1., 0., 0., 1.], variance);
        test_case(Vector2::zeros(), mat2![f64::INFINITY, 0., 0., f64::INFINITY], mat2![f64::INFINITY, 0., 0., f64::INFINITY], variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: MultivariateNormal<U2>| x.entropy().unwrap();
        test_case(Vector2::zeros(), Matrix2::identity(), 2.8378770664093453, entropy);
        test_case(Vector2::zeros(), mat2![1., 0.5, 0.5, 1.], 2.694036030183455, entropy);
        test_case(Vector2::zeros(), mat2![f64::INFINITY, 0., 0., f64::INFINITY], f64::INFINITY, entropy);
    }

    #[test]
    fn test_mode() {
        let mode = |x: MultivariateNormal<U2>| x.mode();
        test_case(Vector2::zeros(), Matrix2::identity(), vect2![0.,  0.], mode);
        test_case(Vector2::repeat(f64::INFINITY), Matrix2::identity(), vect2![f64::INFINITY,  f64::INFINITY], mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: MultivariateNormal<U2>| x.min();
        let max = |x: MultivariateNormal<U2>| x.max();
        test_case(Vector2::zeros(), Matrix2::identity(), vec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(Vector2::zeros(), Matrix2::identity(), vec![f64::INFINITY, f64::INFINITY], max);
        test_case(vect2![10., 1.], Matrix2::identity(), vec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(vect2![-3., 5.], Matrix2::identity(), vec![f64::INFINITY, f64::INFINITY], max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: Vector2<f64>| move |x: MultivariateNormal<U2>| x.pdf(arg);
        test_case(Vector2::zeros(), Matrix2::identity(), 0.05854983152431917, pdf(vect2![1., 1.]));
        test_almost(Vector2::zeros(), Matrix2::identity(), 0.013064233284684921, 1e-15, pdf(vect2![1., 2.]));
        test_almost(Vector2::zeros(), Matrix2::identity(), 1.8618676045881531e-23, 1e-35, pdf(vect2![1., 10.]));
        test_almost(Vector2::zeros(), Matrix2::identity(), 5.920684802611216e-45, 1e-58, pdf(vect2![10., 10.]));
        test_almost(Vector2::zeros(), mat2![1., 0.9, 0.9, 1.], 1.6576716577547003e-05, 1e-18, pdf(vect2![1., -1.]));
        test_almost(Vector2::zeros(), mat2![1., 0.99, 0.99, 1.], 4.1970621773477824e-44, 1e-54, pdf(vect2![1., -1.]));
        test_almost(vect2![0.5, -0.2], mat2![2.0, 0.3, 0.3,  0.5], 0.0013075203140666656, 1e-15, pdf(vect2![2., 2.]));
        test_case(Vector2::zeros(), mat2![f64::INFINITY, 0., 0., f64::INFINITY], 0.0, pdf(vect2![10., 10.]));
        test_case(Vector2::zeros(), mat2![f64::INFINITY, 0., 0., f64::INFINITY], 0.0, pdf(vect2![100., 100.]));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: Vector2<_>| move |x: MultivariateNormal<U2>| x.ln_pdf(arg);
        test_case(Vector2::zeros(), Matrix2::identity(), (0.05854983152431917f64).ln(), ln_pdf(vect2![1., 1.]));
        test_almost(Vector2::zeros(), Matrix2::identity(), (0.013064233284684921f64).ln(), 1e-15, ln_pdf(vect2![1., 2.]));
        test_almost(Vector2::zeros(), Matrix2::identity(), (1.8618676045881531e-23f64).ln(), 1e-15, ln_pdf(vect2![1., 10.]));
        test_almost(Vector2::zeros(), Matrix2::identity(), (5.920684802611216e-45f64).ln(), 1e-15, ln_pdf(vect2![10., 10.]));
        test_almost(Vector2::zeros(), mat2![1., 0.9, 0.9, 1.], (1.6576716577547003e-05f64).ln(), 1e-14, ln_pdf(vect2![1., -1.]));
        test_almost(Vector2::zeros(), mat2![1., 0.99, 0.99, 1.], (4.1970621773477824e-44f64).ln(), 1e-12, ln_pdf(vect2![1., -1.]));
        test_almost(vect2![0.5, -0.2], mat2![2.0, 0.3, 0.3, 0.5],  (0.0013075203140666656f64).ln(), 1e-15, ln_pdf(vect2![2., 2.]));
        test_case(Vector2::zeros(), mat2![f64::INFINITY, 0., 0., f64::INFINITY], f64::NEG_INFINITY, ln_pdf(vect2![10., 10.]));
        test_case(Vector2::zeros(), mat2![f64::INFINITY, 0., 0., f64::INFINITY], f64::NEG_INFINITY, ln_pdf(vect2![100., 100.]));
    }
}
