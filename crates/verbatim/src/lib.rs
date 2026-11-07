use bytemuck::{AnyBitPattern, NoUninit, cast_slice, cast_slice_mut};

use std::{
	io::{
		BufRead, Error as IoError, ErrorKind, Read, Result as IoResult,
		Write,
	},
	mem,
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
) -> IoResult<()> {
	let bytes = reader.fill_buf()?;
	if bytes.starts_with(tag) {
		reader.consume(tag.len());
		Ok(())
	} else {
		Err(IoError::new(
			ErrorKind::InvalidData,
			format!("expected tag: {tag:?}"),
		))
	}
}

pub struct VerbatimWriter<W> {
	writer: W,
	written: usize,
}

pub struct VerbatimReader<R> {
	reader: R,
	read: usize,
}

macro_rules! writer_primitive_le {
	($type:ty, $size:literal, $write:ident) => {
		pub fn $write(&mut self, value: $type) -> IoResult<()> {
			let r = self.written % $size;
			if r != 0 {
				const PAD: [u8; $size] = [0; $size];
				self.writer.write_all(&PAD[..$size - r])?;
				self.written += $size - r;
			}
			self.writer.write_all(&value.to_le_bytes())?;
			self.written += $size;
			Ok(())
		}
	};
}

impl<W: Write> VerbatimWriter<W> {
	pub fn new(writer: W) -> Self {
		Self { writer, written: 0 }
	}

	pub fn into_inner(self) -> W {
		self.writer
	}

	pub fn write_u8(&mut self, value: u8) -> IoResult<()> {
		self.writer.write_all(&[value])?;
		self.written += 1;
		Ok(())
	}

	pub fn write_i8(&mut self, value: i8) -> IoResult<()> {
		self.write_u8(value as u8)
	}

	writer_primitive_le!(u16, 2, write_u16);
	writer_primitive_le!(u32, 4, write_u32);
	writer_primitive_le!(u64, 8, write_u64);
	writer_primitive_le!(u128, 16, write_u128);
	writer_primitive_le!(i16, 2, write_i16);
	writer_primitive_le!(i32, 4, write_i32);
	writer_primitive_le!(i64, 8, write_i64);
	writer_primitive_le!(i128, 16, write_i128);
	writer_primitive_le!(f32, 4, write_f32);
	writer_primitive_le!(f64, 8, write_f64);

	pub fn write_bool(&mut self, value: bool) -> IoResult<()> {
		self.write_u8(u8::from(value))
	}

	pub fn write_slice<T>(&mut self, slice: &[T]) -> IoResult<()>
	where
		T: NoUninit,
	{
		self.write_u32(slice.len() as u32)?;
		const { assert!(mem::align_of::<T>() <= 32) }
		let align = mem::align_of::<T>();
		let r = self.written % align;
		if r != 0 && !slice.is_empty() {
			const PAD: [u8; 32] = [0; 32];
			self.writer.write_all(&PAD[..align - r])?;
			self.written += align - r;
		}
		let bytes: &[u8] = cast_slice(slice);
		self.writer.write_all(bytes)?;
		self.written += bytes.len();
		Ok(())
	}
}

macro_rules! reader_primitive_le {
	($type:ty, $size:literal, $read:ident) => {
		pub fn $read(&mut self) -> IoResult<$type> {
			let r = self.read % $size;
			if r != 0 {
				let mut pad = [0u8; $size];
				let skipped = self
					.reader
					.read(&mut pad[..$size - r])?;
				if skipped != $size - r {
					return Err(IoError::new(
						ErrorKind::UnexpectedEof,
						"unexpected end while reading padding",
					));
				}
				self.read += $size - r;
			}
			let mut buf = [0u8; $size];
			self.reader.read_exact(&mut buf)?;
			self.read += $size;
			Ok(<$type>::from_le_bytes(buf))
		}
	};
}

impl<R: Read + BufRead> VerbatimReader<R> {
	pub fn new(reader: R) -> Self {
		Self { reader, read: 0 }
	}

	pub fn into_inner(self) -> R {
		self.reader
	}

	pub fn read_u8(&mut self) -> IoResult<u8> {
		let mut buf = [0u8; 1];
		self.reader.read_exact(&mut buf)?;
		self.read += 1;
		Ok(buf[0])
	}

	pub fn read_i8(&mut self) -> IoResult<i8> {
		Ok(self.read_u8()? as i8)
	}

	reader_primitive_le!(u16, 2, read_u16);
	reader_primitive_le!(u32, 4, read_u32);
	reader_primitive_le!(u64, 8, read_u64);
	reader_primitive_le!(u128, 16, read_u128);
	reader_primitive_le!(i16, 2, read_i16);
	reader_primitive_le!(i32, 4, read_i32);
	reader_primitive_le!(i64, 8, read_i64);
	reader_primitive_le!(i128, 16, read_i128);
	reader_primitive_le!(f32, 4, read_f32);
	reader_primitive_le!(f64, 8, read_f64);

	pub fn read_bool(&mut self) -> IoResult<bool> {
		match self.read_u8()? {
			0 => Ok(false),
			1 => Ok(true),
			value => Err(IoError::new(
				ErrorKind::InvalidData,
				format!("invalid boolean value: {value:x}"),
			)),
		}
	}

	pub fn read_slice<T>(&mut self) -> IoResult<Vec<T>>
	where
		T: NoUninit + AnyBitPattern,
	{
		let len = self.read_u32()? as usize;
		const { assert!(mem::align_of::<T>() <= 32) }
		let align = mem::align_of::<T>();
		let r = self.read % align;
		if r != 0 && len != 0 {
			let mut pad = [0u8; 32];
			let skipped = self.reader.read(&mut pad[..align - r])?;
			if skipped != align - r {
				return Err(IoError::new(
					ErrorKind::UnexpectedEof,
					"unexpected end while reading padding",
				));
			}
			self.read += align - r;
		}

		let mut values = vec![T::zeroed(); len];
		let bytes: &mut [u8] = cast_slice_mut(&mut values);
		self.reader.read_exact(bytes)?;
		self.read += bytes.len();
		Ok(values)
	}
}
