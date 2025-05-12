use serde::ser::{Serialize, SerializeSeq, SerializeStruct};

use crate::SkVec;

impl<T: Serialize> Serialize for SkVec<T> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		let mut state = serializer.serialize_struct("SkVec", 3)?;
		state.serialize_field("edited", &self.edited)?;
		state.serialize_field("mask", &self.mask)?;
		state.serialize_field("inner", &Inner(&self))?;
		state.end()
	}
}

#[repr(transparent)]
struct Inner<'a, T>(&'a SkVec<T>);

impl<'a, T: Serialize> Serialize for Inner<'a, T> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		let mut seq =
			serializer.serialize_seq(Some(self.0.len() * 2))?;

		// Uninitialized slots are deserialized as `None`
		for i in 0..self.0.len() {
			seq.serialize_element(&self.0.first_item(i))?;
			seq.serialize_element(&self.0.second_item(i))?;
		}

		seq.end()
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use serde_json;

	#[test]
	fn serialize() {
		let mut vec = SkVec::new();
		vec.push(1);
		vec.push(2);
		vec.push(3);
		vec.set(1, 20);

		assert_eq!(
			r#"{"edited":[false,true,false],"mask":[0,1,0],"inner":[1,null,2,20,3,null]}"#,
			serde_json::to_string(&vec).unwrap()
		);
	}
}
