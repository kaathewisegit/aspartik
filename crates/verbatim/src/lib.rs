use bytemuck::{AnyBitPattern, NoUninit, cast_slice};

use std::{
	cmp,
	io::{Error as IoError, ErrorKind, Read, Result as IoResult, Write},
	mem,
};

pub struct VerbatimWriter<W> {
	writer: W,
	written: usize,
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

/// [`VerbatimWriter`] backed by a borrowed [`Write`] trait object
pub type DynVerbatimWriter<'a> = VerbatimWriter<&'a mut (dyn Write + 'a)>;

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

	pub fn write_tag<const N: usize>(
		&mut self,
		tag: [u8; N],
	) -> IoResult<()> {
		self.writer.write_all(&tag)?;
		self.written += N;
		Ok(())
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

pub struct VerbatimReader<R, const BUFFER_SIZE: usize = 8192> {
	reader: R,
	buffer: Vec<u8>,
	start: usize,
	read: usize,
}

macro_rules! reader_primitive_le {
	($type:ty, $size:literal, $read:ident) => {
		pub fn $read(&mut self) -> IoResult<$type> {
			let r = self.read % $size;
			if r != 0 {
				let pad = $size - r;
				self.demand(pad)?;
				self.start += pad;
				self.read += $size - r;
			}
			self.demand($size)?;
			let buf: [u8; $size] =
				self.peek($size).try_into().unwrap();
			self.start += $size;
			self.read += $size;
			Ok(<$type>::from_le_bytes(buf))
		}
	};
}

impl<R: Read> VerbatimReader<R, 8192> {
	pub fn new(reader: R) -> Self {
		Self {
			reader,
			buffer: Vec::with_capacity(8192),
			start: 0,
			read: 0,
		}
	}
}

impl<R: Read, const BUFFER_SIZE: usize> VerbatimReader<R, BUFFER_SIZE> {
	pub fn with_capacity(reader: R) -> Self {
		Self {
			reader,
			buffer: Vec::with_capacity(BUFFER_SIZE),
			start: 0,
			read: 0,
		}
	}

	pub fn into_inner(self) -> R {
		self.reader
	}

	/// Extend the buffer, relocating active bytes it to the start
	fn fill(&mut self, required: usize) -> IoResult<usize> {
		let active = self.buffer.len() - self.start;
		self.buffer.copy_within(self.start.., 0);
		self.start = 0;
		let target = cmp::max(required, BUFFER_SIZE);
		if self.buffer.len() < target {
			self.buffer.resize(target, 0);
		}
		let read = self.reader.read(&mut self.buffer[active..])?;
		self.buffer.truncate(active + read);
		Ok(read)
	}

	/// Read at least `required` bytes into the buffer
	fn demand(&mut self, required: usize) -> IoResult<()> {
		while self.buffer.len() - self.start < required {
			if self.fill(required)? == 0 {
				return Err(IoError::new(
					ErrorKind::UnexpectedEof,
					"unexpected end while filling the buffer",
				));
			}
		}
		Ok(())
	}

	fn peek(&self, len: usize) -> &[u8] {
		&self.buffer[self.start..self.start + len]
	}

	pub fn read_u8(&mut self) -> IoResult<u8> {
		self.demand(1)?;
		let value = self.buffer[self.start];
		self.read += 1;
		self.start += 1;
		Ok(value)
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

	pub fn read_tag<const N: usize>(
		&mut self,
		tag: [u8; N],
	) -> IoResult<()> {
		self.demand(N)?;
		if self.peek(N) == tag {
			self.start += N;
			self.read += N;
			Ok(())
		} else {
			Err(IoError::new(
				ErrorKind::InvalidData,
				format!("expected tag: {tag:?}"),
			))
		}
	}

	pub fn read_slice<T>(&mut self) -> IoResult<&[T]>
	where
		T: NoUninit + AnyBitPattern,
	{
		let len = self.read_u32()? as usize;
		const { assert!(mem::align_of::<T>() <= 32) }
		let align = mem::align_of::<T>();
		let r = self.read % align;
		if r != 0 && len != 0 {
			let pad = align - r;
			self.demand(pad)?;
			self.start += pad;
			self.read += align - r;
		}

		let byte_len = len * mem::size_of::<T>();
		self.demand(byte_len)?;

		if self.start > 0 {
			let old_len = self.buffer.len();
			self.buffer.copy_within(self.start.., 0);
			self.buffer.resize(old_len - self.start, 0);
			self.start = 0;
		}

		self.start += byte_len;
		self.read += byte_len;
		let bytes = &self.buffer[0..byte_len];
		Ok(cast_slice(bytes))
	}
}
