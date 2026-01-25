use serde::{
	Deserialize, Serialize,
	de::{Error, Visitor},
};

use std::{fmt, marker::PhantomData};

use super::{Character, Sequence};

impl<C: Character> Serialize for Sequence<C> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.serialize_bytes(self.as_bytes())
	}
}

struct SequenceVisitor<C>(PhantomData<C>);

impl<'de, C: Character> Visitor<'de> for SequenceVisitor<C> {
	type Value = Sequence<C>;

	fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
		formatter.write_str("a byte array")
	}

	fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
	where
		E: Error,
	{
		Sequence::copy_from_byte_slice(v)
			.ok_or_else(|| E::custom("wrong byte"))
	}
}

impl<'de, C: Character> Deserialize<'de> for Sequence<C> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		deserializer
			.deserialize_bytes(SequenceVisitor(PhantomData::<C>))
	}
}
