//! Provides functions that don't have a numerical solution and must
//! be solved computationally (e.g. evaluation of a polynomial)

/// evaluates a polynomial at `z` where `coeff` are the coeffecients
/// to a polynomial of order `k` where `k` is the length of `coeff` and the
/// coeffecient
/// to the `k`th power is the `k`th element in coeff. E.g. [3,-1,2] equates to
/// `2z^2 - z + 3`
///
/// # Remarks
///
/// Returns 0 for a 0 length coefficient slice
pub fn polynomial(z: f64, coeff: &[f64]) -> f64 {
	let n = coeff.len();
	if n == 0 {
		return 0.0;
	}

	let mut sum = *coeff.last().unwrap();
	for c in coeff[0..n - 1].iter().rev() {
		sum = *c + z * sum;
	}
	sum
}
