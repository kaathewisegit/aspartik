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
		if on {
			self.set_on(index)
		} else {
			self.set_off(index)
		}
	}

	pub fn set_on(&mut self, index: usize) {
		let byte_index = index / 8;
		let bit_value = 1 << (index % 8);
		self.inner[byte_index] |= bit_value;
	}

	pub fn set_off(&mut self, index: usize) {
		let byte_index = index / 8;
		let mask = !(1 << (index % 8));
		self.inner[byte_index] &= mask;
	}

	pub fn at(&self, index: usize) -> bool {
		let byte_index = index / 8;
		let bit_value = 1 << (index % 8);
		(self.inner[byte_index] & bit_value) != 0
	}

	pub fn set_all_off(&mut self) {
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
