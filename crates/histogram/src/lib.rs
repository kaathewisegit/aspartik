use std::ops::AddAssign;

#[derive(Debug, Default, Clone)]
pub struct Histogram<T> {
	keys: Vec<u32>,
	values: Vec<T>,
}

impl<T> Histogram<T>
where
	T: AddAssign + Copy,
{
	pub fn add(&mut self, key: f64, value: T) {
		let int_key = self.to_int_key(key);

		match self.keys.binary_search(&int_key) {
			Ok(index) => self.values[index] += value,
			Err(insert_index) => {
				self.keys.insert(insert_index, int_key);
				self.values.insert(insert_index, value);
			}
		}
	}

	pub fn get(&self, key: f64) -> Option<T> {
		let int_key = self.to_int_key(key);

		match self.keys.binary_search(&int_key) {
			Ok(index) => Some(self.values[index]),
			Err(_) => None,
		}
	}

	pub fn clear(&mut self) {
		self.keys.clear();
		self.values.clear();
	}

	fn to_int_key(&self, key: f64) -> u32 {
		let rounded_key = key.abs().trunc();
		rounded_key as u32
	}
}
