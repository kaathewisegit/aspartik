#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bitmap {
	inner: Box<[u8]>,
}

impl Bitmap {
	pub fn new(size: usize) -> Self {
		let length = (size + 7) / 8;
		Bitmap {
			inner: (0..length).map(|_| 0).collect(),
		}
	}

	pub fn set(&mut self, index: usize, value: bool) {
		let byte_index = index / 8;
		let bit_value = u8::from(value) << (index % 8);
		self.inner[byte_index] |= bit_value;
	}

	pub fn contains(&self, index: usize) -> bool {
		let byte_index = index / 8;
		let bit_value = 1 << (index % 8);
		(self.inner[byte_index] & bit_value) > 0
	}

	pub fn clear(&mut self) {
		for byte in &mut self.inner {
			*byte = 0;
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

		assert!(b.contains(1));
		assert!(b.contains(2));
		assert!(b.contains(9));
		assert!(!b.contains(0));
		assert!(!b.contains(3));
		assert!(!b.contains(4));
	}
}
