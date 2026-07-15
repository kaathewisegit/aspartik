use bytemuck::{NoUninit, cast_slice};

use std::io::{
	BufRead, Error as IoError, ErrorKind, Result as IoResult, Write,
};

macro_rules! primitive_le {
	($type:ty, $size:literal, $read:ident, $write:ident) => {
		pub fn $read<R: BufRead + ?Sized>(
			reader: &mut R,
		) -> IoResult<$type> {
			let mut buf = [0u8; $size];
			reader.read_exact(&mut buf)?;
			Ok(<$type>::from_le_bytes(buf))
		}

		pub fn $write<W: Write + ?Sized>(
			writer: &mut W,
			value: $type,
		) -> IoResult<()> {
			writer.write_all(&value.to_le_bytes())
		}
	};
}

primitive_le!(u8, 1, read_u8, write_u8);
primitive_le!(u16, 2, read_u16_le, write_u16_le);
primitive_le!(u32, 4, read_u32_le, write_u32_le);
primitive_le!(u64, 8, read_u64_le, write_u64_le);
primitive_le!(u128, 16, read_u128_le, write_u128_le);
primitive_le!(i8, 1, read_i8_le, write_i8_le);
primitive_le!(i16, 2, read_i16_le, write_i16_le);
primitive_le!(i32, 4, read_i32_le, write_i32_le);
primitive_le!(i64, 8, read_i64_le, write_i64_le);
primitive_le!(i128, 16, read_i128_le, write_i128_le);
primitive_le!(f32, 4, read_f32_le, write_f32_le);
primitive_le!(f64, 8, read_f64_le, write_f64_le);

pub fn write_bool<W: Write + ?Sized>(
	writer: &mut W,
	value: bool,
) -> IoResult<()> {
	write_u8(writer, u8::from(value))
}

pub fn read_bool<R: BufRead + ?Sized>(reader: &mut R) -> IoResult<bool> {
	match read_u8(reader)? {
		0 => Ok(false),
		1 => Ok(true),
		value => Err(IoError::new(
			ErrorKind::InvalidData,
			format!("invalid boolean value: {value:x}"),
		)),
	}
}

pub fn write_bytes_len32<W: Write + ?Sized>(
	writer: &mut W,
	bytes: &[u8],
) -> IoResult<()> {
	write_u32_le(writer, bytes.len() as u32)?;
	writer.write_all(bytes)
}

pub fn write_slice_len32<W: Write + ?Sized, T: NoUninit>(
	writer: &mut W,
	slice: &[T],
) -> IoResult<()> {
	write_u32_le(writer, slice.len() as u32)?;
	writer.write_all(cast_slice(slice))
}

pub fn write_tag<W: Write + ?Sized>(
	writer: &mut W,
	tag: &[u8],
) -> IoResult<()> {
	writer.write_all(tag)
}

pub fn read_tag<R: BufRead + ?Sized>(
	reader: &mut R,
	tag: &[u8],
) -> IoResult<bool> {
	let bytes = reader.fill_buf()?;
	Ok(bytes.starts_with(tag))
}
