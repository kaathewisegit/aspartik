use arbtest::arbtest;

use math::float;

#[test]
fn f64_sign() {
	arbtest(|u| {
		let x: f64 = u.arbitrary()?;

		let sign = float::sign(x);
		assert!((sign && x.is_sign_negative())
			|| (!sign && x.is_sign_positive()));

		Ok(())
	});
}

const MANTISSA_MASK: u64 = !u64::MAX << 52;
const EXPONENT_MASK: u16 = !u16::MAX << 11;

#[test]
fn f64_bits() {
	arbtest(|u| {
		let sign: bool = u.arbitrary()?;
		let exponent = u.arbitrary::<u16>()? & EXPONENT_MASK;
		let mantissa = u.arbitrary::<u64>()? & MANTISSA_MASK;

		let sign_u64: u64 = sign.into();
		let exponent_u64: u64 = exponent.into();
		let bits = (sign_u64 << 63) | (exponent_u64 << 51) | mantissa;
		let f = f64::from_bits(bits);

		assert_eq!(float::sign(f), sign);
		assert_eq!(float::exponent_bits(f), exponent);
		assert_eq!(float::mantissa_bits(f), mantissa);

		Ok(())
	});
}
