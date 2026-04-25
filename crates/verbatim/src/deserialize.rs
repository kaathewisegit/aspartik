use anyhow::{Context, Result, bail};

use std::mem;

pub trait Deserialize<'r>: Sized {
	fn deserialize(bytes: &mut &'r [u8]) -> Result<Self>;
}

pub trait DeserializeOwned: for<'r> Deserialize<'r> {}
impl<T> DeserializeOwned for T where T: for<'r> Deserialize<'r> {}

fn read_slice<'r>(bytes: &mut &'r [u8], len: usize) -> Result<&'r [u8]> {
	let Some((read, rest)) = mem::take(bytes).split_at_checked(len) else {
		bail!(
			"Tried to read {} bytes from a slice the size of {}",
			len,
			bytes.len()
		);
	};
	*bytes = rest;
	Ok(read)
}

macro_rules! impl_de_from_bytes {
	($type:ty, $num_bytes:literal) => {
		impl<'r> Deserialize<'r> for $type {
			fn deserialize(bytes: &mut &'r [u8]) -> Result<Self> {
				Ok(<$type>::from_le_bytes(
					*read_slice(bytes, $num_bytes)?
						.as_array::<$num_bytes>()
						.expect(
							"read_slice must return the exact number of bytes",
						),
				))
			}
		}
	};
}

impl_de_from_bytes!(u8, 1);
impl_de_from_bytes!(u16, 2);
impl_de_from_bytes!(u32, 4);
impl_de_from_bytes!(u64, 8);
impl_de_from_bytes!(u128, 16);
impl_de_from_bytes!(i8, 1);
impl_de_from_bytes!(i16, 2);
impl_de_from_bytes!(i32, 4);
impl_de_from_bytes!(i64, 8);
impl_de_from_bytes!(i128, 16);
impl_de_from_bytes!(f32, 4);
impl_de_from_bytes!(f64, 8);

impl<'r> Deserialize<'r> for bool {
	fn deserialize(bytes: &mut &'r [u8]) -> Result<Self> {
		match read_slice(bytes, 1)?[0] {
			0 => Ok(false),
			1 => Ok(true),
			other => bail!("Invalid bool value: {other:x}"),
		}
	}
}

impl<'r> Deserialize<'r> for char {
	fn deserialize(bytes: &mut &'r [u8]) -> Result<Self> {
		let ch = u32::deserialize(bytes)?;
		char::from_u32(ch)
			.with_context(|| format!("Invalid char value: {ch:x}"))
	}
}

impl<'r> Deserialize<'r> for &'r [u8] {
	fn deserialize(bytes: &mut &'r [u8]) -> Result<Self> {
		// TODO: length size?
		let len = u32::deserialize(bytes)? as usize;
		read_slice(bytes, len)
	}
}

impl<'r> Deserialize<'r> for &'r str {
	fn deserialize(bytes: &mut &'r [u8]) -> Result<Self> {
		// TODO: length size?
		let len = u32::deserialize(bytes)? as usize;
		let bytes = read_slice(bytes, len)?;
		Ok(str::from_utf8(bytes)?)
	}
}
