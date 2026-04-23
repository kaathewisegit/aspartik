use anyhow::{Context, Result, bail};

use crate::Read;

pub trait Deserialize<'r>: Sized {
	fn deserialize<R>(reader: &mut R) -> Result<Self>
	where
		R: Read<'r>;
}

pub trait DeserializeOwned: for<'r> Deserialize<'r> {}
impl<T> DeserializeOwned for T where T: for<'r> Deserialize<'r> {}

pub trait DeserializeFrom {
	fn deserialize_from<'r, R>(self, reader: &mut R) -> Result<()>
	where
		R: Read<'r>;
}

macro_rules! impl_de_from_bytes {
	($type:ty, $num_bytes:literal) => {
		impl<'r> Deserialize<'r> for $type {
			fn deserialize<R>(reader: &mut R) -> Result<Self>
			where
				R: Read<'r>,
			{
				Ok(<$type>::from_le_bytes(
					reader.read_array::<$num_bytes>()?,
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
	fn deserialize<R: Read<'r>>(reader: &mut R) -> Result<Self> {
		match reader.read_array::<1>()?[0] {
			0 => Ok(false),
			1 => Ok(true),
			other => bail!("Invalid bool value: {other:x}"),
		}
	}
}

impl<'r> Deserialize<'r> for char {
	fn deserialize<R: Read<'r>>(reader: &mut R) -> Result<Self> {
		let ch = u32::deserialize(reader)?;
		char::from_u32(ch)
			.with_context(|| format!("Invalid char value: {ch:x}"))
	}
}

impl<'r> Deserialize<'r> for &'r [u8] {
	fn deserialize<R: Read<'r>>(reader: &mut R) -> Result<Self> {
		// TODO: length size?
		let len = u32::deserialize(reader)? as usize;
		reader.read_slice(len)
	}
}

impl<'r> Deserialize<'r> for &'r str {
	fn deserialize<R: Read<'r>>(reader: &mut R) -> Result<Self> {
		// TODO: length size?
		let len = u32::deserialize(reader)? as usize;
		let bytes = reader.read_slice(len)?;
		Ok(str::from_utf8(bytes)?)
	}
}
