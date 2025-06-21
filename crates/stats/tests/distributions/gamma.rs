use stats::distribution::{Gamma, GammaError};

use crate::{check_continuous_distribution, prelude::*};

test_new_is_ok! {
	Gamma;
	(1.0, 0.1),
	(1.0, 1.0),
	(10.0, 10.0),
	(10.0, 1.0),
	(10.0, f64::INFINITY),
}

test_new_is_err! {
	Gamma;
	(0.0, 0.0) -> GammaError::ShapeInvalid,
	(1.0, f64::NAN) -> GammaError::RateInvalid,
	(1.0, -1.0) -> GammaError::RateInvalid,
	(-1.0, 1.0) -> GammaError::ShapeInvalid,
	(-1.0, -1.0) -> GammaError::ShapeInvalid,
	(-1.0, f64::NAN) -> GammaError::ShapeInvalid,
	(f64::INFINITY, f64::INFINITY) -> GammaError::ShapeAndRateInfinite,
}

test_value! {
	test_mean, Gamma, mean;
	(1.0, 0.1): () => Some(10.0),
	(1.0, 1.0): () => Some(1.0),
	(10.0, 10.0): () => Some(1.0),
	(10.0, 1.0): () => Some(10.0),
	(10.0, f64::INFINITY): () => Some(0.0),
}

test_value! {
	test_variance, Gamma, variance;
	(1.0, 0.1): () => 100.0, relative unwrap,
	(1.0, 1.0): () => Some(1.0),
	(10.0, 10.0): () => Some(0.1),
	(10.0, 1.0): () => Some(10.0),
	(10.0, f64::INFINITY): () => Some(0.0),
}

test_value! {
	test_entropy, Gamma, entropy;
	(1.0, 0.1): () => 3.302585092994045628506840223, relative unwrap,
	(1.0, 1.0): () => 1.0, relative unwrap,
	(10.0, 10.0): () => 0.2334690854869339583626209, relative unwrap,
	(10.0, 1.0): () => 2.53605417848097964238061239, relative unwrap,
	(10.0, f64::INFINITY): () => Some(f64::NEG_INFINITY),
}

test_value! {
	test_skewness, Gamma, skewness;
	(1.0, 0.1): () => Some(2.0),
	(1.0, 1.0): () => Some(2.0),
	(10.0, 10.0): () => Some(0.6324555320336758663997787),
	(10.0, 1.0): () => Some(0.63245553203367586639977870),
	(10.0, f64::INFINITY): () => 0.6324555320336758, relative unwrap,
}

test_value! {
	test_mode, Gamma, mode;
	(1.0, 0.1): () => Some(0.0),
	(1.0, 1.0): () => Some(0.0),
	(10.0, 10.0): () => Some(0.9),
	(10.0, 1.0): () => Some(9.0),
	(10.0, f64::INFINITY): () => Some(0.0),
}

test_value! {
	test_lower, Gamma, lower;
	(1.0, 0.1): () => 0.0,
	(1.0, 1.0): () => 0.0,
	(10.0, 10.0): () => 0.0,
	(10.0, 1.0): () => 0.0,
	(10.0, f64::INFINITY): () => 0.0,
}

test_value! {
	test_upper, Gamma, upper;
	(1.0, 0.1): () => f64::INFINITY,
	(1.0, 1.0): () => f64::INFINITY,
	(10.0, 10.0): () => f64::INFINITY,
	(10.0, 1.0): () => f64::INFINITY,
	(10.0, f64::INFINITY): () => f64::INFINITY,
}

test_value! {
	test_pdf, Gamma, pdf;
	(1.0, 0.1): (1.0) => 0.090483741803595961836995,
	(1.0, 0.1): (10.0) => 0.036787944117144234201693,
	(1.0, 1.0): (1.0) => 0.367879441171442321595523,
	(1.0, 1.0): (10.0) => 0.000045399929762484851535,
	(10.0, 10.0): (1.0) => 1.251100357211332989847649, relative,
	(10.0, 10.0): (10.0) => 1.025153212086870580621609e-30, relative,
	(10.0, 1.0): (1.0) => 0.000001013777119630297402, relative,
	(10.0, 1.0): (10.0) => 0.125110035721133298984764, relative,

	(1.0, 0.1): (0.0) => (0.1),
}

test_value! {
	test_ln_pdf, Gamma, ln_pdf;
	(1.0, 0.1): (1.0) => -2.40258509299404563405795,
	(1.0, 0.1): (10.0) => -3.30258509299404562850684,
	(1.0, 1.0): (1.0) => -1.0,
	(1.0, 1.0): (10.0) => -10.0,
	(10.0, 10.0): (1.0) => 0.224023449858987228972196, relative,
	(10.0, 10.0): (10.0) => -69.0527107131946016148658, relative,
	(10.0, 1.0): (1.0) => -13.8018274800814696112077, relative,
	(10.0, 1.0): (10.0) => -2.07856164313505845504579, relative,
	(10.0, f64::INFINITY): (f64::INFINITY) => f64::NEG_INFINITY,

	(1.0, 0.1): (0.0) => (0.1f64.ln()),
}

test_value! {
	test_cdf, Gamma, cdf;
	(1.0, 0.1): (1.0) => 0.095162581964040431858607, relative,
	(1.0, 0.1): (10.0) => 0.632120558828557678404476, relative,
	(1.0, 1.0): (1.0) => 0.632120558828557678404476, relative,
	(1.0, 1.0): (10.0) => 0.999954600070237515148464, relative,
	(10.0, 10.0): (1.0) => 0.542070285528147791685835, relative,
	(10.0, 10.0): (10.0) => 0.999999999999999999999999, relative,
	(10.0, 1.0): (1.0) => 0.000000111425478338720677, relative,
	(10.0, 1.0): (10.0) => 0.542070285528147791685835, relative,
	(10.0, f64::INFINITY): (1.0) => 0.0,
	(10.0, f64::INFINITY): (10.0) => 1.0,

	(1.0, 0.1): (0.0) => (0.0),
}

test_value! {
	test_sf, Gamma, sf;
	(1.0, 0.1): (1.0) => 0.9048374180359595,
	(1.0, 0.1): (10.0) => 0.3678794411714419,
	(1.0, 1.0): (1.0) => 0.3678794411714419, relative,
	(1.0, 1.0): (10.0) => 4.539992976249074e-5, relative,
	(10.0, 10.0): (1.0) => 0.4579297144718528, relative,
	(10.0, 10.0): (10.0) => 1.1253473960842808e-31, relative,
	(10.0, 1.0): (1.0) => 0.9999998885745217, relative,
	(10.0, 1.0): (10.0) => 0.4579297144718528, relative,
	(10.0, f64::INFINITY): (1.0) => 1.0,
	(10.0, f64::INFINITY): (10.0) => 0.0,

	(1.0, 0.1): (0.0) => (1.0),
}

#[test]
fn test_continuous() {
	check_continuous_distribution(
		&Gamma::new(1.0, 0.5).unwrap(),
		0.0,
		20.0,
	);
	check_continuous_distribution(
		&Gamma::new(9.0, 2.0).unwrap(),
		0.0,
		20.0,
	);
}

// #[test]
// fn test_cdf_inverse_identity() {
// 	let f = |p: f64| move |g: Gamma| g.cdf(g.inverse_cdf(p));
// 	let params = [
// 		(1.0, 0.1),
// 		(1.0, 1.0),
// 		(10.0, 10.0),
// 		(10.0, 1.0),
// 		(100.0, 200.0),
// 	];
//
// 	for (s, r) in params {
// 		for n in -5..0 {
// 			let p = 10.0f64.powi(n);
// 			test_relative(s, r, p, f(p));
// 		}
// 	}
//
// 	// test case from issue #200
// 	{
// 		let x = 20.5567;
// 		let f = |x: f64| move |g: Gamma| g.inverse_cdf(g.cdf(x));
// 		test_relative(3.0, 0.5, x, f(x))
// 	}
// }
