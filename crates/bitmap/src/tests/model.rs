use rand::{Rng as RngTrait, SeedableRng};
use rand_pcg::Pcg64 as Rng;

use crate::Bitmap;

struct Model(Vec<bool>);

impl Model {
	pub fn new(size: usize) -> Self {
		Model(vec![false; size])
	}

	pub fn set(&mut self, index: usize, on: bool) {
		if on {
			self.set_on(index)
		} else {
			self.set_off(index)
		}
	}

	pub fn set_on(&mut self, index: usize) {
		self.0[index] = true;
	}

	pub fn set_off(&mut self, index: usize) {
		self.0[index] = false;
	}

	pub fn at(&self, index: usize) -> bool {
		self.0[index]
	}

	pub fn set_all_off(&mut self) {
		for value in &mut self.0 {
			*value = false;
		}
	}

	pub fn set_all_on(&mut self) {
		for value in &mut self.0 {
			*value = true;
		}
	}

	fn len(&self) -> usize {
		self.0.len()
	}
}

fn assert_same(bitmap: &Bitmap, model: &Model) {
	for i in 0..model.len() {
		assert_eq!(bitmap.at(i), model.at(i));
	}
}

fn run(length: usize, steps: usize) {
	let mut rng = Rng::seed_from_u64(4);

	let mut model = Model::new(length);
	let mut bitmap = Bitmap::new(length);

	for _ in 0..steps {
		let op = rng.random_range(0..=76);

		match op {
			0..25 => {
				let index = rng.random_range(0..length);
				model.set_on(index);
				bitmap.set_on(index);
			}
			25..50 => {
				let index = rng.random_range(0..length);
				model.set_off(index);
				bitmap.set_off(index);
			}
			50..75 => {
				let index = rng.random_range(0..length);
				let state: bool = rng.random();

				model.set(index, state);
				bitmap.set(index, state);
			}
			75 => {
				model.set_all_on();
				bitmap.set_all_on();
			}
			76 => {
				model.set_all_off();
				bitmap.set_all_off();
			}
			_ => unreachable!(),
		}

		assert_same(&bitmap, &model);
	}
}

#[test]
fn single() {
	run(1, 100_000);
}

#[test]
fn short8() {
	run(8, 100_000);
}

#[test]
fn short32() {
	run(32, 100_000);
}

#[test]
fn short_non_8() {
	run(7, 100_000);
}

#[test]
fn non_8() {
	run(77, 100_000);
}

#[test]
fn long() {
	run(999, 10_000);
}
