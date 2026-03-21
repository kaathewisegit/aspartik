use stats::distribution::{FisherSnedecor, FisherSnedecorError};

use crate::prelude::*;

make_test_harness!(
    FisherSnedecor(freedom_1: f64, freedom_2: f64),
    FisherSnedecorError
);

#[test]
fn test_new_is_ok() {
	let cases = [
		(0.1, 0.1),
		(1.0, 0.1),
		(10.0, 0.1),
		(0.1, 1.0),
		(1.0, 1.0),
		(10.0, 1.0),
	];
	for args in cases {
		assert_new_is_ok(args);
	}
}

#[test]
fn test_new_is_err() {
	let cases = [
		((f64::INFINITY, 0.1), FisherSnedecorError::Freedom1Invalid),
		((0.1, f64::INFINITY), FisherSnedecorError::Freedom2Invalid),
		((f64::NAN, f64::NAN), FisherSnedecorError::Freedom1Invalid),
		((0.0, f64::NAN), FisherSnedecorError::Freedom1Invalid),
		((-1.0, f64::NAN), FisherSnedecorError::Freedom1Invalid),
		((-10.0, f64::NAN), FisherSnedecorError::Freedom1Invalid),
		((f64::NAN, 0.0), FisherSnedecorError::Freedom1Invalid),
		((0.0, 0.0), FisherSnedecorError::Freedom1Invalid),
		((-1.0, 0.0), FisherSnedecorError::Freedom1Invalid),
		((-10.0, 0.0), FisherSnedecorError::Freedom1Invalid),
		((f64::NAN, -1.0), FisherSnedecorError::Freedom1Invalid),
		((0.0, -1.0), FisherSnedecorError::Freedom1Invalid),
		((-1.0, -1.0), FisherSnedecorError::Freedom1Invalid),
		((-10.0, -1.0), FisherSnedecorError::Freedom1Invalid),
		((f64::NAN, -10.0), FisherSnedecorError::Freedom1Invalid),
		((0.0, -10.0), FisherSnedecorError::Freedom1Invalid),
		((-1.0, -10.0), FisherSnedecorError::Freedom1Invalid),
		((-10.0, -10.0), FisherSnedecorError::Freedom1Invalid),
		(
			(f64::INFINITY, f64::INFINITY),
			FisherSnedecorError::Freedom1Invalid,
		),
	];
	for (args, err) in cases {
		assert_new_is_err(args, err);
	}
}

#[test]
fn test_mean() {
	let cases = [
		((0.1, 10.0), 1.25),
		((1.0, 10.0), 1.25),
		((10.0, 10.0), 1.25),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mean().unwrap(), expected);
	}
}

#[test]
fn test_mean_with_low_d2() {
	let cases = [((0.1, 0.1), ())];
	for (args, _) in cases {
		assert!(new_dist(args).mean().is_none());
	}
}

#[test]
fn test_variance() {
	let cases = [
		((0.1, 10.0), 42.1875),
		((1.0, 10.0), 4.6875),
		((10.0, 10.0), 0.9375),
	];
	for (args, expected) in cases {
		let variance = new_dist(args).variance().unwrap();
		assert_almost_eq!(variance, expected, relative = 1e-14);
	}
}

#[test]
fn test_variance_with_low_d2() {
	let cases = [((0.1, 0.1), ())];
	for (args, _) in cases {
		assert!(new_dist(args).variance().is_none());
	}
}

#[test]
fn test_skewness() {
	let cases = [
		((0.1, 10.0), 15.78090735784977),
		((1.0, 10.0), 5.773502691896257),
		((10.0, 10.0), 3.6147844564602556),
	];
	for (args, expected) in cases {
		let skewness = new_dist(args).skewness().unwrap();
		assert_almost_eq!(skewness, expected, relative = 1e-14);
	}
}

#[test]
fn test_skewness_with_low_d2() {
	let cases = [((0.1, 0.1), ())];
	for (args, _) in cases {
		assert!(new_dist(args).skewness().is_none());
	}
}

#[test]
fn test_mode() {
	let cases = [
		((10.0, 0.1), 0.0380952380952381),
		((10.0, 1.0), 4.0 / 15.0),
		((10.0, 10.0), 2.0 / 3.0),
	];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.mode().unwrap(), expected);
	}
}

#[test]
fn test_mode_with_low_d1() {
	let cases = [((0.1, 0.1), ())];
	for (args, _) in cases {
		assert!(new_dist(args).mode().is_none());
	}
}

#[test]
fn test_lower() {
	let cases = [((1.0, 1.0), 0.0)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.lower(), expected);
	}
}

#[test]
fn test_upper() {
	let cases = [((1.0, 1.0), f64::INFINITY)];
	for (args, expected) in cases {
		assert_exact(args, (), |d, _| d.upper(), expected);
	}
}

#[test]
fn test_pdf() {
	let cases = [
		((0.1, 0.1), 1.0, 0.023415420722658897),
		((1.0, 0.1), 1.0, 0.039606456091066396),
		((10.0, 0.1), 1.0, 0.04184406304005453),
		((0.1, 1.0), 1.0, 0.039606456091066396),
		((1.0, 1.0), 1.0, 0.15915494309189535),
		((10.0, 1.0), 1.0, 0.23036198922913864),
		((0.1, 0.1), 10.0, 0.00221546909694001),
		((1.0, 0.1), 10.0, 0.0036996037038792263),
		((10.0, 0.1), 10.0, 0.0039017972117414293),
		((0.1, 1.0), 10.0, 0.0031986407335993154),
		((1.0, 1.0), 10.0, 0.009150765837179461),
		((10.0, 1.0), 10.0, 0.011649385917144215),
		((0.1, 10.0), 10.0, 0.00305087016058574),
		((1.0, 10.0), 10.0, 0.0027189774911347956),
		((10.0, 10.0), 10.0, 2.42892272340605E-4),
	];
	for (args, p, expected) in cases {
		let pdf = new_dist(args).pdf(p);
		assert_almost_eq!(pdf, expected, relative = 1e-13);
	}
}

#[test]
fn test_ln_pdf() {
	let cases = [
		((0.1, 0.1), 1.0, 0.023415420722658897_f64.ln()),
		((1.0, 0.1), 1.0, 0.039606456091066396_f64.ln()),
		((10.0, 0.1), 1.0, 0.04184406304005453_f64.ln()),
		((0.1, 1.0), 1.0, 0.039606456091066396_f64.ln()),
		((1.0, 1.0), 1.0, 0.15915494309189535_f64.ln()),
		((10.0, 1.0), 1.0, 0.23036198922913864_f64.ln()),
		((0.1, 0.1), 10.0, 0.00221546909694001_f64.ln()),
		((1.0, 0.1), 10.0, 0.0036996037038792263_f64.ln()),
		((10.0, 0.1), 10.0, 0.0039017972117414293_f64.ln()),
		((0.1, 1.0), 10.0, 0.0031986407335993154_f64.ln()),
		((1.0, 1.0), 10.0, 0.009150765837179461_f64.ln()),
		((10.0, 1.0), 10.0, 0.011649385917144215_f64.ln()),
		((0.1, 10.0), 10.0, 0.00305087016058574_f64.ln()),
		((1.0, 10.0), 10.0, 0.0027189774911347956_f64.ln()),
		((10.0, 10.0), 10.0, 2.42892272340605E-4_f64.ln()),
	];
	for (args, p, expected) in cases {
		let ln_pdf = new_dist(args).ln_pdf(p);
		assert_almost_eq!(ln_pdf, expected, relative = 1e-13);
	}
}

#[test]
fn test_cdf() {
	let cases = [
		((0.1, 0.1), 0.1, 0.4471298603342514),
		((1.0, 0.1), 0.1, 0.08156522095104674),
		((10.0, 0.1), 0.1, 0.033184005716276534),
		((0.1, 1.0), 0.1, 0.7437871091798638),
		((1.0, 1.0), 0.1, 0.19498222904213663),
		((10.0, 1.0), 0.1, 0.010119559735433714),
		((0.1, 0.1), 1.0, 0.5),
		((1.0, 0.1), 1.0, 0.16734351500944272),
		((10.0, 0.1), 1.0, 0.12207560664741705),
		((0.1, 1.0), 1.0, 0.8326564849905573),
		((1.0, 1.0), 1.0, 0.5),
		((10.0, 1.0), 1.0, 0.34089313230205986),
	];
	for (args, p, expected) in cases {
		let cdf = new_dist(args).cdf(p);
		assert_almost_eq!(cdf, expected, relative = 1e-11);
	}
}

#[test]
fn test_cdf_lower_bound() {
	let cases = [((0.1, 0.1), -1.0, 0.0)];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.cdf(p), expected);
	}
}

#[test]
fn test_sf() {
	let cases = [
		((0.1, 0.1), 0.1, 0.5528701396657489),
		((1.0, 0.1), 0.1, 0.9184347790489533),
		((10.0, 0.1), 0.1, 0.9668159942836896),
		((0.1, 1.0), 0.1, 0.25621289082013654),
		((1.0, 1.0), 0.1, 0.8050177709578634),
		((10.0, 1.0), 0.1, 0.9898804402645662),
		((0.1, 0.1), 1.0, 0.5),
		((1.0, 0.1), 1.0, 0.8326564849905562),
		((10.0, 0.1), 1.0, 0.8779243933525519),
		((0.1, 1.0), 1.0, 0.16734351500944344),
		((1.0, 1.0), 1.0, 0.5),
		((10.0, 1.0), 1.0, 0.65910686769794),
	];
	for (args, p, expected) in cases {
		let sf = new_dist(args).sf(p);
		assert_almost_eq!(sf, expected, relative = 1e-13);
	}
}

#[test]
fn test_inverse_cdf() {
	let cases = [
		((0.1, 0.1), 0.1, 0.1),
		((1.0, 0.1), 0.1, 0.1),
		((10.0, 0.1), 0.1, 0.1),
		((0.1, 1.0), 0.1, 0.1),
		((1.0, 1.0), 0.1, 0.1),
		((10.0, 1.0), 0.1, 0.1),
		((0.1, 0.1), 1.0, 1.0),
		((1.0, 0.1), 1.0, 1.0),
		((10.0, 0.1), 1.0, 1.0),
		((0.1, 1.0), 1.0, 1.0),
		((1.0, 1.0), 1.0, 1.0),
		((10.0, 1.0), 1.0, 1.0),
	];
	for (args, p, expected) in cases {
		let dist = new_dist(args);
		let prob = Probability::new(dist.cdf(p));
		let inverse_cdf = dist.inverse_cdf(prob);
		assert_almost_eq!(inverse_cdf, expected, relative = 1e-12);
	}
}

#[test]
fn test_sf_lower_bound() {
	let cases = [((0.1, 0.1), -1.0, 1.0)];
	for (args, p, expected) in cases {
		assert_exact(args, p, |d, p| d.sf(p), expected);
	}
}

#[test]
fn test_continuous() {
	let cases = [((10.0, 10.0), 0.0, 10.0)];
	for (args, lower, upper) in cases {
		check_continuous_distribution(&new_dist(args), lower, upper);
	}
}
