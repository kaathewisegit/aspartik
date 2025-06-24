/// `sqrt(2 * pi)`
pub const SQRT_2PI: f64 = 2.5066282746310007;

/// `ln(pi)`
pub const LN_PI: f64 = 1.1447298858494002;

/// `ln(sqrt(2 * pi))`
pub const LN_SQRT_2PI: f64 = 0.9189385332046728;

/// `ln(sqrt(2 * pi * e))`
pub const LN_SQRT_2PIE: f64 = 1.4189385332046727;

/// `ln(2 * sqrt(e / pi))`
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452;

/// `2 * sqrt(e / pi)`
pub const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657;

/// Euler-Masheroni constant: `lim(n -> inf) { sum(k=1 -> n) { 1/k - ln(n) } }`
pub const EULER_MASCHERONI: f64 = 0.5772156649015329;

pub use core::f64::consts::*;
