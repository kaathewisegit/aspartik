use anyhow::Result;

pub trait Parser<Input: ?Sized> {
	type Output;

	fn advance(
		&mut self,
		input: &mut &Input,
	) -> Result<Option<Self::Output>>;

	fn final_object(&mut self) -> Option<Self::Output>;
}
