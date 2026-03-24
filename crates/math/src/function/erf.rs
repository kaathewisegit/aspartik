//! Provides the [error](https://en.wikipedia.org/wiki/Error_function) and
//! related functions

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::function::evaluate;

/// Error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erf(x: f64) -> f64 {
	libm::erf(x)
}

/// Inverse error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erf_inv(x: f64) -> f64 {
	if x == 0.0 {
		0.0
	} else if x >= 1.0 {
		f64::INFINITY
	} else if x <= -1.0 {
		f64::NEG_INFINITY
	} else if x < 0.0 {
		erf_inv_impl(-x, 1.0 + x, -1.0)
	} else {
		erf_inv_impl(x, 1.0 - x, 1.0)
	}
}

/// Complementary error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erfc(x: f64) -> f64 {
	libm::erfc(x)
}

/// Complementary inverse error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erfc_inv(x: f64) -> f64 {
	if x <= 0.0 {
		f64::INFINITY
	} else if x >= 2.0 {
		f64::NEG_INFINITY
	} else if x > 1.0 {
		erf_inv_impl(-1.0 + x, 2.0 - x, -1.0)
	} else {
		erf_inv_impl(1.0 - x, x, 1.0)
	}
}

// **********************************************************
// ********** Coefficients for erf_inv_impl polynomial ******
// **********************************************************

/// Polynomial coefficients for a numerator of `erf_inv_impl`
/// in the interval [0, 0.5].
const ERF_INV_IMPL_AN: &[f64] = &[
	-0.0005087819496582806,
	-0.008368748197417368,
	0.03348066254097446,
	-0.012692614766297404,
	-0.03656379714117627,
	0.02198786811111689,
	0.008226878746769157,
	-0.005387729650712429,
];

/// Polynomial coefficients for a denominator of `erf_inv_impl`
/// in the interval [0, 0.5].
const ERF_INV_IMPL_AD: &[f64] = &[
	1.0,
	-0.9700050433032906,
	-1.5657455823417585,
	1.5622155839842302,
	0.662328840472003,
	-0.7122890234154284,
	-0.05273963823400997,
	0.07952836873415717,
	-0.0023339375937419,
	0.0008862163904564247,
];

/// Polynomial coefficients for a numerator of `erf_inv_impl`
/// in the interval [0.5, 0.75].
const ERF_INV_IMPL_BN: &[f64] = &[
	-0.20243350835593876,
	0.10526468069939171,
	8.3705032834312,
	17.644729840837403,
	-18.851064805871424,
	-44.6382324441787,
	17.445385985570866,
	21.12946554483405,
	-3.6719225470772936,
];

/// Polynomial coefficients for a denominator of `erf_inv_impl`
/// in the interval [0.5, 0.75].
const ERF_INV_IMPL_BD: &[f64] = &[
	1.0,
	6.242641248542475,
	3.971343795334387,
	-28.66081804998,
	-20.14326346804852,
	48.560921310873994,
	10.826866735546016,
	-22.643693341313973,
	1.7211476576120028,
];

/// Polynomial coefficients for a numerator of `erf_inv_impl`
/// in the interval [0.75, 1] with x less than 3.
const ERF_INV_IMPL_CN: &[f64] = &[
	-0.1311027816799519,
	-0.16379404719331705,
	0.11703015634199525,
	0.38707973897260434,
	0.3377855389120359,
	0.14286953440815717,
	0.029015791000532906,
	0.0021455899538880526,
	-6.794655751811263e-7,
	2.8522533178221704e-8,
	-6.81149956853777e-10,
];

/// Polynomial coefficients for a denominator of `erf_inv_impl`
/// in the interval [0.75, 1] with x less than 3.
const ERF_INV_IMPL_CD: &[f64] = &[
	1.0,
	3.4662540724256723,
	5.381683457070069,
	4.778465929458438,
	2.5930192162362027,
	0.848854343457902,
	0.15226433829533179,
	0.011059242293464892,
];

/// Polynomial coefficients for a numerator of `erf_inv_impl`
/// in the interval [0.75, 1] with x between 3 and 6.
const ERF_INV_IMPL_DN: &[f64] = &[
	-0.0350353787183178,
	-0.0022242652921344794,
	0.018557330651423107,
	0.009508047013259196,
	0.0018712349281955923,
	0.00015754461742496055,
	4.60469890584318e-6,
	-2.304047769118826e-10,
	2.6633922742578204e-12,
];

/// Polynomial coefficients for a denominator of `erf_inv_impl`
/// in the interval [0.75, 1] with x between 3 and 6.
const ERF_INV_IMPL_DD: &[f64] = &[
	1.0,
	1.3653349817554064,
	0.7620591645536234,
	0.22009110576413124,
	0.03415891436709477,
	0.00263861676657016,
	7.646752923027944e-5,
];

/// Polynomial coefficients for a numerator of `erf_inv_impl`
/// in the interval [0.75, 1] with x between 6 and 18.
const ERF_INV_IMPL_EN: &[f64] = &[
	-0.016743100507663373,
	-0.0011295143874558028,
	0.001056288621524929,
	0.00020938631748758808,
	1.4962478375834237e-5,
	4.4969678992770644e-7,
	4.625961635228786e-9,
	-2.811287356288318e-14,
	9.905570997331033e-17,
];

/// Polynomial coefficients for a denominator of `erf_inv_impl`
/// in the interval [0.75, 1] with x between 6 and 18.
const ERF_INV_IMPL_ED: &[f64] = &[
	1.0,
	0.5914293448864175,
	0.1381518657490833,
	0.016074608709367652,
	0.0009640118070051656,
	2.7533547476472603e-5,
	2.82243172016108e-7,
];

/// Polynomial coefficients for a numerator of `erf_inv_impl`
/// in the interval [0.75, 1] with x between 18 and 44.
const ERF_INV_IMPL_FN: &[f64] = &[
	-0.002497821279189813,
	-7.79190719229054e-6,
	2.5472303741302746e-5,
	1.6239777734251093e-6,
	3.963410113048012e-8,
	4.116328311909442e-10,
	1.4559628671867504e-12,
	-1.1676501239718427e-18,
];

/// Polynomial coefficients for a denominator of `erf_inv_impl`
/// in the interval [0.75, 1] with x between 18 and 44.
const ERF_INV_IMPL_FD: &[f64] = &[
	1.0,
	0.2071231122144225,
	0.01694108381209759,
	0.0006905382656226846,
	1.4500735981823264e-5,
	1.4443775662814415e-7,
	5.097612765997785e-10,
];

/// Polynomial coefficients for a numerator of `erf_inv_impl`
/// in the interval [0.75, 1] with x greater than 44.
const ERF_INV_IMPL_GN: &[f64] = &[
	-0.0005390429110190785,
	-2.8398759004727723e-7,
	8.994651148922914e-7,
	2.2934585926592085e-8,
	2.2556144486350015e-10,
	9.478466275030226e-13,
	1.3588013010892486e-15,
	-3.4889039339994887e-22,
];

/// Polynomial coefficients for a denominator of `erf_inv_impl`
/// in the interval [0.75, 1] with x greater than 44.
const ERF_INV_IMPL_GD: &[f64] = &[
	1.0,
	0.08457462340018994,
	0.002820929847262647,
	4.682929219408942e-5,
	3.999688121938621e-7,
	1.6180929088790448e-9,
	2.315586083102596e-12,
];

// `erf_inv_impl` computes the inverse error function where
// `p`,`q`, and `s` are the first, second, and third intermediate
// parameters respectively
fn erf_inv_impl(p: f64, q: f64, s: f64) -> f64 {
	let result = if p <= 0.5 {
		let y = 0.08913147449493408;
		let g = p * (p + 10.0);
		let r = evaluate::polynomial(p, ERF_INV_IMPL_AN)
			/ evaluate::polynomial(p, ERF_INV_IMPL_AD);
		g * y + g * r
	} else if q >= 0.25 {
		let y = 2.249481201171875;
		let g = (-2.0 * q.ln()).sqrt();
		let xs = q - 0.25;
		let r = evaluate::polynomial(xs, ERF_INV_IMPL_BN)
			/ evaluate::polynomial(xs, ERF_INV_IMPL_BD);
		g / (y + r)
	} else {
		let x = (-q.ln()).sqrt();
		if x < 3.0 {
			let y = 0.807220458984375;
			let xs = x - 1.125;
			let r = evaluate::polynomial(xs, ERF_INV_IMPL_CN)
				/ evaluate::polynomial(xs, ERF_INV_IMPL_CD);
			y * x + r * x
		} else if x < 6.0 {
			let y = 0.9399557113647461;
			let xs = x - 3.0;
			let r = evaluate::polynomial(xs, ERF_INV_IMPL_DN)
				/ evaluate::polynomial(xs, ERF_INV_IMPL_DD);
			y * x + r * x
		} else if x < 18.0 {
			let y = 0.9836282730102539;
			let xs = x - 6.0;
			let r = evaluate::polynomial(xs, ERF_INV_IMPL_EN)
				/ evaluate::polynomial(xs, ERF_INV_IMPL_ED);
			y * x + r * x
		} else if x < 44.0 {
			let y = 0.9971456527709961;
			let xs = x - 18.0;
			let r = evaluate::polynomial(xs, ERF_INV_IMPL_FN)
				/ evaluate::polynomial(xs, ERF_INV_IMPL_FD);
			y * x + r * x
		} else {
			let y = 0.9994134902954102;
			let xs = x - 44.0;
			let r = evaluate::polynomial(xs, ERF_INV_IMPL_GN)
				/ evaluate::polynomial(xs, ERF_INV_IMPL_GD);
			y * x + r * x
		}
	};
	s * result
}
