//! Provides the [error](https://en.wikipedia.org/wiki/Error_function) and
//! related functions

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::function::evaluate;

/// Error function
#[cfg_attr(feature = "python", pyfunction)]
pub fn erf(x: f64) -> f64 {
	if x.is_nan() {
		f64::NAN
	} else if x >= 0.0 && x.is_infinite() {
		1.0
	} else if x <= 0.0 && x.is_infinite() {
		-1.0
	} else if x == 0.0 {
		0.0
	} else {
		erf_impl(x, false)
	}
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
	if x.is_nan() {
		f64::NAN
	} else if x == f64::INFINITY {
		0.0
	} else if x == f64::NEG_INFINITY {
		2.0
	} else {
		erf_impl(x, true)
	}
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

// Coefficients for erf_impl polynomial

/// Polynomial coefficients for a numerator of `erf_impl`
/// in the interval [1e-10, 0.5].
const ERF_IMPL_AN: &[f64] = &[
	0.0033791670955125737,
	-0.0007369565304816795,
	-0.3747323373929196,
	0.08174424487335873,
	-0.04210893199365486,
	0.007016570951209575,
	-0.004950912559824351,
	0.0008716465990379225,
];

/// Polynomial coefficients for a denominator of `erf_impl`
/// in the interval [1e-10, 0.5]
const ERF_IMPL_AD: &[f64] = &[
	1.0,
	-0.21808821808792464,
	0.4125429727254421,
	-0.08418911478731067,
	0.06553388564002416,
	-0.012001960445494177,
	0.00408165558926174,
	-0.0006159007215577697,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [0.5, 0.75].
const ERF_IMPL_BN: &[f64] = &[
	-0.03617903907182625,
	0.2922518834448827,
	0.2814470417976045,
	0.12561020886276694,
	0.027413502826893053,
	0.0025083967216806575,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [0.5, 0.75].
const ERF_IMPL_BD: &[f64] = &[
	1.0,
	1.8545005897903486,
	1.4357580303783142,
	0.5828276587530365,
	0.12481047693294975,
	0.011372417654635328,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [0.75, 1.25].
const ERF_IMPL_CN: &[f64] = &[
	-0.03978768926111369,
	0.1531652124678783,
	0.19126029560093624,
	0.10276327061989304,
	0.029637090615738836,
	0.004609348678027549,
	0.0003076078203486802,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [0.75, 1.25].
const ERF_IMPL_CD: &[f64] = &[
	1.0,
	1.955200729876277,
	1.6476231719938486,
	0.7682386070221262,
	0.20979318593650978,
	0.031956931689991336,
	0.0021336316089578537,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [1.25, 2.25].
const ERF_IMPL_DN: &[f64] = &[
	-0.030083856055794972,
	0.05385788298444545,
	0.07262115416519142,
	0.036762846988804936,
	0.009646290155725275,
	0.0013345348007529107,
	7.780875997825043e-5,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [1.25, 2.25].
const ERF_IMPL_DD: &[f64] = &[
	1.0,
	1.7596709814716753,
	1.3288357143796112,
	0.5525285965087576,
	0.13379305694133287,
	0.017950964517628076,
	0.0010471244001993736,
	-1.0664038182035734e-8,
];

///  Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [2.25, 3.5].
const ERF_IMPL_EN: &[f64] = &[
	-0.011790757013722784,
	0.01426213209053881,
	0.020223443590296084,
	0.009306682999904321,
	0.00213357802422066,
	0.00025022987386460105,
	1.2053491221958819e-5,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [2.25, 3.5].
const ERF_IMPL_ED: &[f64] = &[
	1.0,
	1.5037622520362048,
	0.9653977862044629,
	0.3392652304767967,
	0.06897406495415698,
	0.007710602624917683,
	0.0003714211015310693,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [3.5, 5.25].
const ERF_IMPL_FN: &[f64] = &[
	-0.005469547955387293,
	0.004041902787317071,
	0.005496336955316117,
	0.002126164726039454,
	0.0003949840144950839,
	3.655654770644424e-5,
	1.3548589710993232e-6,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [3.5, 5.25].
const ERF_IMPL_FD: &[f64] = &[
	1.0,
	1.2101969777363077,
	0.6209146682211439,
	0.17303843066114277,
	0.027655081377343203,
	0.0024062597442430973,
	8.918118172513365e-5,
	-4.655288362833827e-12,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [5.25, 8].
const ERF_IMPL_GN: &[f64] = &[
	-0.0027072253590577837,
	0.00131875634250294,
	0.0011992593326100233,
	0.00027849619811344664,
	2.6782298821833186e-5,
	9.230436723150282e-7,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [5.25, 8].
const ERF_IMPL_GD: &[f64] = &[
	1.0,
	0.8146328085431416,
	0.26890166585629954,
	0.044987721610304114,
	0.0038175966332024847,
	0.00013157189788859692,
	4.048153596757641e-12,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [8, 11.5].
const ERF_IMPL_HN: &[f64] = &[
	-0.001099467206917422,
	0.00040642544275042267,
	0.0002744994894169007,
	4.652937706466594e-5,
	3.2095542539576746e-6,
	7.782860181450209e-8,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [8, 11.5].
const ERF_IMPL_HD: &[f64] = &[
	1.0,
	0.5881737106118461,
	0.13936333128940975,
	0.016632934041708368,
	0.0010002392131023491,
	2.4254837521587224e-5,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [11.5, 17].
const ERF_IMPL_IN: &[f64] = &[
	-0.0005690799360109496,
	0.00016949854037376225,
	5.184723545811009e-5,
	3.8281931223192885e-6,
	8.249899312818944e-8,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [11.5, 17].
const ERF_IMPL_ID: &[f64] = &[
	1.0,
	0.33963725005113937,
	0.04347264787031066,
	0.002485493352246371,
	5.356333053371529e-5,
	-1.1749094440545958e-13,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [17, 24].
const ERF_IMPL_JN: &[f64] = &[
	-0.00024131359948399134,
	5.742249752025015e-5,
	1.1599896292738377e-5,
	5.817621344025938e-7,
	8.539715550856736e-9,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [17, 24].
const ERF_IMPL_JD: &[f64] = &[
	1.0,
	0.23304413829968784,
	0.02041869405464403,
	0.0007971856475643983,
	1.1701928167017232e-5,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [24, 38].
const ERF_IMPL_KN: &[f64] = &[
	-0.00014667469927776036,
	1.6266655211228053e-5,
	2.6911624850916523e-6,
	9.79584479468092e-8,
	1.0199464762572346e-9,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [24, 38].
const ERF_IMPL_KD: &[f64] = &[
	1.0,
	0.16590781294484722,
	0.010336171619150588,
	0.0002865930263738684,
	2.9840157084090034e-6,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [38, 60].
const ERF_IMPL_LN: &[f64] = &[
	-5.839057976297718e-5,
	4.125103251054962e-6,
	4.3179092242025094e-7,
	9.933651555900132e-9,
	6.534805100201047e-11,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [38, 60].
const ERF_IMPL_LD: &[f64] = &[
	1.0,
	0.10507708607203992,
	0.004142784286754756,
	7.263387546445238e-5,
	4.778184710473988e-7,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [60, 85].
const ERF_IMPL_MN: &[f64] = &[
	-1.9645779760922958e-5,
	1.572438876668007e-6,
	5.439025111927009e-8,
	3.174724923691177e-10,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [60, 85].
const ERF_IMPL_MD: &[f64] = &[
	1.0,
	0.05280398924095763,
	0.0009268760691517533,
	5.410117232266303e-6,
	5.350938458036424e-16,
];

/// Polynomial coefficients for a numerator in `erf_impl`
/// in the interval [85, 110].
const ERF_IMPL_NN: &[f64] = &[
	-7.892247039787227e-6,
	6.22088451660987e-7,
	1.457284456768824e-8,
	6.037155055427153e-11,
];

/// Polynomial coefficients for a denominator in `erf_impl`
/// in the interval [85, 110].
const ERF_IMPL_ND: &[f64] = &[
	1.0,
	0.03753288463562937,
	0.0004679195359746253,
	1.9384703927584565e-6,
];

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

/// `erf_impl` computes the error function at `z`.
/// If `inv` is true, `1 - erf` is calculated as opposed to `erf`
fn erf_impl(z: f64, inv: bool) -> f64 {
	if z < 0.0 {
		if !inv {
			return -erf_impl(-z, false);
		}
		if z < -0.5 {
			return 2.0 - erf_impl(-z, true);
		}
		return 1.0 + erf_impl(-z, false);
	}

	let result = if z < 0.5 {
		if z < 1e-10 {
			z * 1.125 + z * 0.0033791670955125737
		} else {
			z * 1.125
				+ z * evaluate::polynomial(z, ERF_IMPL_AN)
					/ evaluate::polynomial(z, ERF_IMPL_AD)
		}
	} else if z < 110.0 {
		let (r, b) = if z < 0.75 {
			(
				evaluate::polynomial(z - 0.5, ERF_IMPL_BN)
					/ evaluate::polynomial(
						z - 0.5,
						ERF_IMPL_BD,
					),
				0.3440242112,
			)
		} else if z < 1.25 {
			(
				evaluate::polynomial(z - 0.75, ERF_IMPL_CN)
					/ evaluate::polynomial(
						z - 0.75,
						ERF_IMPL_CD,
					),
				0.419990927,
			)
		} else if z < 2.25 {
			(
				evaluate::polynomial(z - 1.25, ERF_IMPL_DN)
					/ evaluate::polynomial(
						z - 1.25,
						ERF_IMPL_DD,
					),
				0.4898625016,
			)
		} else if z < 3.5 {
			(
				evaluate::polynomial(z - 2.25, ERF_IMPL_EN)
					/ evaluate::polynomial(
						z - 2.25,
						ERF_IMPL_ED,
					),
				0.5317370892,
			)
		} else if z < 5.25 {
			(
				evaluate::polynomial(z - 3.5, ERF_IMPL_FN)
					/ evaluate::polynomial(
						z - 3.5,
						ERF_IMPL_FD,
					),
				0.5489973426,
			)
		} else if z < 8.0 {
			(
				evaluate::polynomial(z - 5.25, ERF_IMPL_GN)
					/ evaluate::polynomial(
						z - 5.25,
						ERF_IMPL_GD,
					),
				0.5571740866,
			)
		} else if z < 11.5 {
			(
				evaluate::polynomial(z - 8.0, ERF_IMPL_HN)
					/ evaluate::polynomial(
						z - 8.0,
						ERF_IMPL_HD,
					),
				0.5609807968,
			)
		} else if z < 17.0 {
			(
				evaluate::polynomial(z - 11.5, ERF_IMPL_IN)
					/ evaluate::polynomial(
						z - 11.5,
						ERF_IMPL_ID,
					),
				0.5626493692,
			)
		} else if z < 24.0 {
			(
				evaluate::polynomial(z - 17.0, ERF_IMPL_JN)
					/ evaluate::polynomial(
						z - 17.0,
						ERF_IMPL_JD,
					),
				0.5634598136,
			)
		} else if z < 38.0 {
			(
				evaluate::polynomial(z - 24.0, ERF_IMPL_KN)
					/ evaluate::polynomial(
						z - 24.0,
						ERF_IMPL_KD,
					),
				0.5638477802,
			)
		} else if z < 60.0 {
			(
				evaluate::polynomial(z - 38.0, ERF_IMPL_LN)
					/ evaluate::polynomial(
						z - 38.0,
						ERF_IMPL_LD,
					),
				0.5640528202,
			)
		} else if z < 85.0 {
			(
				evaluate::polynomial(z - 60.0, ERF_IMPL_MN)
					/ evaluate::polynomial(
						z - 60.0,
						ERF_IMPL_MD,
					),
				0.5641309023,
			)
		} else {
			(
				evaluate::polynomial(z - 85.0, ERF_IMPL_NN)
					/ evaluate::polynomial(
						z - 85.0,
						ERF_IMPL_ND,
					),
				0.5641584396,
			)
		};
		let g = (-z * z).exp() / z;
		g * b + g * r
	} else {
		0.0
	};

	if inv && z >= 0.5 {
		result
	} else if z >= 0.5 || inv {
		1.0 - result
	} else {
		result
	}
}

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
