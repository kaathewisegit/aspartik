use math::{assert_almost_eq, function::erf::*};

#[test]
fn test_erf_inf() {
	assert!(erf(f64::NAN).is_nan());
	assert_eq!(erf(f64::INFINITY), 1.0);
	assert_eq!(erf(f64::NEG_INFINITY), -1.0);
}

#[test]
fn test_erf() {
	let cases = [
		(-1.0, -0.8427007929497149, 1e-11),
		(0.0, 0.0, 1e-16),
		(1e-15, 0.0000000000000011283791670955127, 1e-30),
		(0.1, 0.1124629160182849, 1e-16),
		(0.2, 0.22270258921047847, 1e-16),
		(0.3, 0.3286267594591274, 1e-16),
		(0.4, 0.42839235504666845, 1e-16),
		(0.5, 0.5204998778130465, 1e-9),
		(1.0, 0.8427007929497149, 1e-11),
		(1.5, 0.9661051464753108, 1e-11),
		(2.0, 0.9953222650189527, 1e-11),
		(2.5, 0.999593047982555, 1e-13),
		(3.0, 0.9999779095030014, 1e-11),
		(4.0, 0.9999999845827421, 1e-16),
		(5.0, 0.9999999999984626, 1e-16),
		(
			6.0,
			0.99999999999999997848026328750108688340664960081261537,
			1e-16,
		),
	];

	for (x, expected, epsilon) in cases {
		assert_almost_eq!(erf(x), expected, epsilon = epsilon);
	}
}

#[test]
fn test_erfc_inf() {
	assert!(erfc(f64::NAN).is_nan());
	assert_eq!(erfc(f64::INFINITY), 0.0);
	assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
}

#[test]
fn test_erfc() {
	let cases = [
		(-1.0, 1.8427007929497148, 1e-11),
		(0.0, 1.0, 1e-16),
		(0.1, 0.887537083981715, 1e-15),
		(0.2, 0.7772974107895215, 1e-16),
		(0.3, 0.6713732405408726, 1e-16),
		(0.4, 0.5716076449533315, 1e-15),
		(0.5, 0.4795001221869535, 1e-9),
		(1.0, 0.15729920705028513, 1e-11),
		(1.5, 0.033894853524689274, 1e-11),
		(2.0, 0.004677734981047266, 1e-11),
		(2.5, 0.0004069520174449589, 1e-13),
		(3.0, 0.00002209049699858544, 1e-11),
		(4.0, 0.00000001541725790028002, 1e-18),
		(5.0, 0.000000000001537459794428035, 1e-22),
		(6.0, 2.1519736712498913e-17, 1e-26),
		(10.0, 2.088487583762545e-45, 1e-55),
		(15.0, 7.212994172451207e-100, 1e-109),
		(20.0, 5.395865611607901e-176, 1e-186),
		(30.0, 0.0, 1e-16),
		(50.0, 0.0, 1e-16),
		(80.0, 0.0, 1e-16),
	];

	for (x, expected, epsilon) in cases {
		assert_almost_eq!(erfc(x), expected, epsilon = epsilon);
	}
}

#[test]
fn test_erf_inv_special() {
	assert_eq!(erf_inv(0.0), 0.0);
	assert!(erf_inv(f64::NAN).is_nan());
	assert_eq!(erf_inv(-1.0), f64::NEG_INFINITY);
	assert_eq!(erf_inv(1.0), f64::INFINITY);
	assert_eq!(erf_inv(f64::INFINITY), f64::INFINITY);
	assert_eq!(erf_inv(f64::NEG_INFINITY), f64::NEG_INFINITY);
}

#[test]
fn test_erf_inv() {
	let cases = [
		(1e-15, 8.86226925452758e-16, 1e-30),
		(0.1, 0.08885599049425769, 1e-17),
		(0.2, 0.17914345462129166, 1e-15),
		(0.3, 0.2724627147267544, 1e-17),
		(0.4, 0.37080715859355795, 1e-17),
		(0.5, 0.4769362762044699, 1e-17),
	];

	for (x, expected, epsilon) in cases {
		assert_almost_eq!(erf_inv(x), expected, epsilon = epsilon);
	}
}
#[test]
fn test_erfc_inv_special() {
	assert_eq!(erfc_inv(0.0), f64::INFINITY);
	assert_eq!(erfc_inv(2.0), f64::NEG_INFINITY);
}

#[test]
fn test_erfc_inv() {
	let cases = [
		(1e-100, 15.065574702593, 1e-11),
		(1e-30, 8.1486162231699, 1e-12),
		(1e-20, 6.6015806223551, 1e-13),
		(1e-10, 4.572824958544925, 1e-7),
		(1e-5, 3.123413274341571, 1e-11),
		(0.1, 1.1630871536766743, 1e-14),
		(0.2, 0.9061938024368233, 1e-15),
		(0.5, 0.4769362762044699, 1e-17),
		(1.0, 0.0, 1e-16),
		(1.5, -0.4769362762044699, 1e-16),
	];

	for (x, expected, epsilon) in cases {
		assert_almost_eq!(erfc_inv(x), expected, epsilon = epsilon);
	}
}
