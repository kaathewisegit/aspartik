use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bitmap {
	inner: Box<[u8]>,
}

impl Bitmap {
	pub fn new(size: usize) -> Self {
		let length = size.div_ceil(8);
		Bitmap {
			inner: (0..length).map(|_| 0).collect(),
		}
	}

	pub fn set(&mut self, index: usize, on: bool) {
		let byte_index = index / 8;
		let bit_value = u8::from(on) << (index % 8);
		self.inner[byte_index] |= bit_value;
	}

	pub fn set_on(&mut self, index: usize) {
		self.set(index, true)
	}

	pub fn set_off(&mut self, index: usize) {
		self.set(index, false)
	}

	pub fn at(&self, index: usize) -> bool {
		let byte_index = index / 8;
		let bit_value = 1 << (index % 8);
		(self.inner[byte_index] & bit_value) > 0
	}

	pub fn clear(&mut self) {
		for byte in &mut self.inner {
			*byte = 0;
		}
	}

	pub fn set_all_on(&mut self) {
		for byte in &mut self.inner {
			*byte = 0b1111_1111;
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn basic() {
		let mut b = Bitmap::new(10);
		b.set(1, true);
		b.set(2, true);
		b.set(9, true);

		assert!(b.at(1));
		assert!(b.at(2));
		assert!(b.at(9));
		assert!(!b.at(0));
		assert!(!b.at(3));
		assert!(!b.at(4));
	}
}
