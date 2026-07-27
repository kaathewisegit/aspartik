use arbitrary::Arbitrary;
use arbtest::arbtest;

use std::{
	fmt::Debug,
	ops::{Deref, DerefMut},
};

use sk::EpochBuf;

struct Model<T> {
	values: Vec<T>,
	backup: Vec<T>,
}

impl<T> Deref for Model<T> {
	type Target = [T];

	fn deref(&self) -> &[T] {
		&self.values
	}
}

impl<T> DerefMut for Model<T> {
	fn deref_mut(&mut self) -> &mut [T] {
		&mut self.values
	}
}

impl<T: Copy> Model<T> {
	fn repeat(value: T, len: usize) -> Self {
		Self {
			values: vec![value; len],
			backup: vec![value; len],
		}
	}

	fn accept(&mut self) {
		self.backup.copy_from_slice(&self.values)
	}

	fn reject(&mut self) {
		self.values.copy_from_slice(&self.backup)
	}
}

fn test<T>(len: usize, steps: usize)
where
	T: Copy + Default + Debug + PartialEq + for<'a> Arbitrary<'a>,
{
	arbtest(|u| {
		let mut buf = EpochBuf::repeat(T::default(), len);
		let mut model = Model::repeat(T::default(), len);

		for _ in 0..steps {
			match u.arbitrary::<u8>()? {
				0..5 => {
					buf.accept();
					model.accept();
				}
				5..10 => {
					buf.reject();
					model.reject();
				}

				10.. => {
					let index = u.choose_index(len)?;
					let value = u.arbitrary::<T>()?;
					buf[index] = value;
					model[index] = value;
				}
			}
			assert_eq!(buf.as_ref(), model.as_ref());
		}

		Ok(())
	})
	.size_min(64_000);
}

#[test]
fn i64_1000() {
	test::<i64>(1000, 10000);
}

#[test]
fn u8_1000() {
	test::<u8>(1000, 10000);
}

#[test]
fn bool() {
	test::<bool>(1000, 10000);
}
