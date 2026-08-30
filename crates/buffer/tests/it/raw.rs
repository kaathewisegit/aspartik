#![expect(clippy::undocumented_unsafe_blocks)]

use buffer::RawBuffer;

#[test]
fn ptr() {
	let len: usize = 10;
	let mut buf = unsafe { RawBuffer::<i32>::uninit(len) };
	let start = buf.ptr().as_ptr();
	for i in 0..len {
		let expected = unsafe { start.add(i) };
		assert_eq!(unsafe { buf.get(i) }, expected);
		assert_eq!(unsafe { buf.get_mut(i) as *const i32 }, expected);
	}
	unsafe { buf.deallocate(len) };
}

#[test]
fn zeroed() {
	unsafe {
		let len: usize = 10;
		let mut buf = RawBuffer::<i32>::zeroed(len);
		assert_eq!(buf.as_slice(len), &[0; 10]);
		buf.deallocate(len);
	}
}

#[test]
fn reallocate_smaller() {
	unsafe {
		let len_0: usize = 10;
		let mut buf = RawBuffer::<usize>::uninit(len_0);
		for i in 0..len_0 {
			*buf.get_mut(i) = i;
		}

		let len_1: usize = 5;
		buf.reallocate(len_0, len_1);
		assert_eq!(buf.as_slice(len_1), &[0, 1, 2, 3, 4]);

		buf.deallocate(len_1);
	}
}

#[test]
fn reallocate_larger() {
	unsafe {
		let len_0: usize = 3;
		let mut buf = RawBuffer::<usize>::uninit(len_0);
		for i in 0..len_0 {
			*buf.get_mut(i) = i;
		}

		let len_1: usize = 5;
		buf.reallocate(len_0, len_1);
		for i in len_0..len_1 {
			*buf.get_mut(i) = 10 * i;
		}
		assert_eq!(buf.as_slice(len_1), &[0, 1, 2, 30, 40]);

		buf.deallocate(len_1);
	}
}

// TODO: simulation testing
