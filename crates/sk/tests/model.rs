use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64 as Rng;

use sk::SkBuf;

struct Model<T> {
	values: Box<[T]>,
	backup: Box<[T]>,
}

impl<T: Copy> Model<T> {
	fn repeat(value: T, len: usize) -> Self {
		Self {
			values: vec![value; len].into(),
			backup: vec![value; len].into(),
		}
	}

	fn set(&mut self, index: usize, value: T) {
		self.values[index] = value;
	}

	fn accept(&mut self) {
		self.backup.copy_from_slice(&self.values);
	}

	fn reject(&mut self) {
		self.values.copy_from_slice(&self.backup);
	}

	fn as_slice(&self) -> &[T] {
		&self.values
	}
}

fn model_test(len: usize) {
	let mut skbuf = SkBuf::repeat(0, len);
	let mut model = Model::repeat(0, len);

	let mut rng = Rng::seed_from_u64(4);

	for _ in 0..1_000 {
		for _ in 0..100 {
			let index = rng.random_range(0..len);
			let value = rng.random::<i32>();

			skbuf.set(index, value);
			model.set(index, value);
		}

		assert_eq!(skbuf, model.as_slice());

		if rng.random_bool(0.5) {
			skbuf.accept();
			model.accept();
		} else {
			skbuf.reject();
			model.reject();
		}

		assert_eq!(skbuf, model.as_slice());
	}
}

#[test]
fn edits_10() {
	model_test(10);
}

#[test]
fn edits_100() {
	model_test(100);
}

#[test]
fn edits_1000() {
	model_test(100);
}
