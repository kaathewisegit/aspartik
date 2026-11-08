use arbitrary::{Arbitrary, Unstructured};
use arbtest::arbtest;

use verbatim::{VerbatimReader, VerbatimWriter};

macro_rules! padding_test {
	($name:ident, $write:ident, $size:expr, $type:ty) => {
		#[test]
		fn $name() {
			let value = <$type>::from_le_bytes([0x1d; $size]);

			let mut writer = VerbatimWriter::new(Vec::new());
			writer.$write(value).unwrap();
			writer.write_u8(7).unwrap();
			writer.$write(value).unwrap();

			let le = value.to_le_bytes();
			let mut expected = le.to_vec();
			expected.push(7);
			expected.extend([0; $size - 1]);
			expected.extend_from_slice(&le);

			assert_eq!(writer.into_inner(), expected);
		}
	};
}

padding_test!(u8_padding, write_u8, 1, u8);
padding_test!(u16_padding, write_u16, 2, u16);
padding_test!(u32_padding, write_u32, 4, u32);
padding_test!(u64_padding, write_u64, 8, u64);
padding_test!(u128_padding, write_u128, 16, u128);
padding_test!(i8_padding, write_i8, 1, i8);
padding_test!(i16_padding, write_i16, 2, i16);
padding_test!(i32_padding, write_i32, 4, i32);
padding_test!(i64_padding, write_i64, 8, i64);
padding_test!(i128_padding, write_i128, 16, i128);
padding_test!(f32_padding, write_f32, 4, f32);
padding_test!(f64_padding, write_f64, 8, f64);

#[test]
fn tag() {
	const TAG: [u8; 4] = *b"TAG!";

	let mut writer = VerbatimWriter::new(Vec::new());
	writer.write_tag(TAG).unwrap();
	assert_eq!(writer.into_inner(), TAG);

	let mut reader = VerbatimReader::new(TAG.as_slice());
	reader.read_tag(TAG).unwrap();
	assert!(reader.into_inner().is_empty());

	let mut reader = VerbatimReader::new(b"nope".as_slice());
	assert!(reader.read_tag(TAG).is_err());
}

#[test]
fn slice() {
	arbtest(|u: &mut Unstructured<'_>| {
		let slice = Vec::<u16>::arbitrary(u)?;

		let mut writer = VerbatimWriter::new(Vec::new());
		writer.write_u8(7).unwrap();
		writer.write_slice(&slice).unwrap();
		writer.write_u8(8).unwrap();

		let mut expected = vec![7u8];
		expected.extend([0; 3]);
		expected.extend((slice.len() as u32).to_le_bytes());
		expected.extend(slice.iter().flat_map(|v| v.to_le_bytes()));
		expected.push(8);

		assert_eq!(writer.into_inner(), expected);
		Ok(())
	});
}

#[derive(Debug, PartialEq)]
enum Value {
	U8(u8),
	U16(u16),
	U32(u32),
	U64(u64),
	U128(u128),
	I8(i8),
	I16(i16),
	I32(i32),
	I64(i64),
	I128(i128),
	F32(f32),
	F64(f64),
	F32Slice(Vec<f32>),
	F64Slice(Vec<f64>),
}

impl<'a> Arbitrary<'a> for Value {
	fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
		Ok(match u.int_in_range(0..=13)? {
			0 => Self::U8(u8::arbitrary(u)?),
			1 => Self::U16(u16::arbitrary(u)?),
			2 => Self::U32(u32::arbitrary(u)?),
			3 => Self::U64(u64::arbitrary(u)?),
			4 => Self::U128(u128::arbitrary(u)?),
			5 => Self::I8(i8::arbitrary(u)?),
			6 => Self::I16(i16::arbitrary(u)?),
			7 => Self::I32(i32::arbitrary(u)?),
			8 => Self::I64(i64::arbitrary(u)?),
			9 => Self::I128(i128::arbitrary(u)?),
			10 => Self::F32(f32::arbitrary(u)?),
			11 => Self::F64(f64::arbitrary(u)?),
			12 => Self::F32Slice(Vec::<f32>::arbitrary(u)?),
			13 => Self::F64Slice(Vec::<f64>::arbitrary(u)?),
			_ => unreachable!(),
		})
	}
}

impl Value {
	fn write(&self, writer: &mut VerbatimWriter<Vec<u8>>) {
		match self {
			Self::U8(v) => writer.write_u8(*v).unwrap(),
			Self::U16(v) => writer.write_u16(*v).unwrap(),
			Self::U32(v) => writer.write_u32(*v).unwrap(),
			Self::U64(v) => writer.write_u64(*v).unwrap(),
			Self::U128(v) => writer.write_u128(*v).unwrap(),
			Self::I8(v) => writer.write_i8(*v).unwrap(),
			Self::I16(v) => writer.write_i16(*v).unwrap(),
			Self::I32(v) => writer.write_i32(*v).unwrap(),
			Self::I64(v) => writer.write_i64(*v).unwrap(),
			Self::I128(v) => writer.write_i128(*v).unwrap(),
			Self::F32(v) => writer.write_f32(*v).unwrap(),
			Self::F64(v) => writer.write_f64(*v).unwrap(),
			Self::F32Slice(v) => writer.write_slice(v).unwrap(),
			Self::F64Slice(v) => writer.write_slice(v).unwrap(),
		}
	}

	fn same(&self, other: &Self) -> bool {
		match (self, other) {
			(Self::F32(a), Self::F32(b)) => {
				a.to_bits() == b.to_bits()
			}
			(Self::F64(a), Self::F64(b)) => {
				a.to_bits() == b.to_bits()
			}
			(Self::F32Slice(a), Self::F32Slice(b)) => {
				a.len() == b.len()
					&& a.iter().zip(b.iter()).all(
						|(a, b)| {
							a.to_bits()
								== b.to_bits()
						},
					)
			}
			(Self::F64Slice(a), Self::F64Slice(b)) => {
				a.len() == b.len()
					&& a.iter().zip(b.iter()).all(
						|(a, b)| {
							a.to_bits()
								== b.to_bits()
						},
					)
			}
			(a, b) => a == b,
		}
	}
}

fn read(value: &Value, reader: &mut VerbatimReader<&[u8]>) -> Value {
	match value {
		Value::U8(_) => Value::U8(reader.read_u8().unwrap()),
		Value::U16(_) => Value::U16(reader.read_u16().unwrap()),
		Value::U32(_) => Value::U32(reader.read_u32().unwrap()),
		Value::U64(_) => Value::U64(reader.read_u64().unwrap()),
		Value::U128(_) => Value::U128(reader.read_u128().unwrap()),
		Value::I8(_) => Value::I8(reader.read_i8().unwrap()),
		Value::I16(_) => Value::I16(reader.read_i16().unwrap()),
		Value::I32(_) => Value::I32(reader.read_i32().unwrap()),
		Value::I64(_) => Value::I64(reader.read_i64().unwrap()),
		Value::I128(_) => Value::I128(reader.read_i128().unwrap()),
		Value::F32(_) => Value::F32(reader.read_f32().unwrap()),
		Value::F64(_) => Value::F64(reader.read_f64().unwrap()),
		Value::F32Slice(_) => Value::F32Slice(
			reader.read_slice::<f32>().unwrap().to_vec(),
		),
		Value::F64Slice(_) => Value::F64Slice(
			reader.read_slice::<f64>().unwrap().to_vec(),
		),
	}
}

#[test]
fn roundtrip() {
	arbtest(|u: &mut Unstructured<'_>| {
		let len = u.int_in_range(0..=2048)?;
		let mut values = Vec::with_capacity(len);
		for _ in 0..len {
			values.push(Value::arbitrary(u)?);
		}

		let mut writer = VerbatimWriter::new(Vec::new());
		for value in &values {
			value.write(&mut writer);
		}
		let bytes = writer.into_inner();

		let mut reader = VerbatimReader::new(bytes.as_slice());
		let read_values: Vec<_> =
			values.iter().map(|v| read(v, &mut reader)).collect();

		assert!(
			values.len() == read_values.len()
				&& values
					.iter()
					.zip(&read_values)
					.all(|(a, b)| a.same(b)),
			"roundtrip mismatch:\nwritten: {values:?}\nread:    {read_values:?}",
		);
		Ok(())
	})
	.size_min(2u32.pow(20));
}
