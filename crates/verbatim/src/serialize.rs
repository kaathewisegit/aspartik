use anyhow::Result;

use crate::Write;

pub trait Serialize {
	fn serialize<W>(&self, writer: &mut W) -> Result<()>
	where
		W: Write;
}

macro_rules! impl_le_bytes {
	($type:ty) => {
		impl Serialize for $type {
			fn serialize<W: Write>(
				&self,
				writer: &mut W,
			) -> Result<()> {
				writer.write_array(self.to_le_bytes())
			}
		}
	};
}

impl_le_bytes!(u8);
impl_le_bytes!(u16);
impl_le_bytes!(u32);
impl_le_bytes!(u64);
impl_le_bytes!(u128);
impl_le_bytes!(i8);
impl_le_bytes!(i16);
impl_le_bytes!(i32);
impl_le_bytes!(i64);
impl_le_bytes!(i128);
impl_le_bytes!(f32);
impl_le_bytes!(f64);

impl Serialize for bool {
	fn serialize<W: Write>(&self, writer: &mut W) -> Result<()> {
		u8::from(*self).serialize(writer)
	}
}

impl Serialize for char {
	fn serialize<W: Write>(&self, writer: &mut W) -> Result<()> {
		u32::from(*self).serialize(writer)
	}
}

impl Serialize for &[u8] {
	fn serialize<W: Write>(&self, writer: &mut W) -> Result<()> {
		(self.len() as u32).serialize(writer)?;
		writer.write_slice(self)
	}
}

impl Serialize for &str {
	fn serialize<W: Write>(&self, writer: &mut W) -> Result<()> {
		(self.len() as u32).serialize(writer)?;
		writer.write_slice(self.as_bytes())
	}
}
