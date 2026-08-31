#![expect(clippy::undocumented_unsafe_blocks)]

use arbtest::arbtest;

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

fn from_slice<const A: usize>() {
	arbtest(|u| unsafe {
		let vec = u.arbitrary::<Vec<i32>>()?;
		if vec.is_empty() {
			return Ok(());
		}
		let mut raw = RawBuffer::<i32, A>::from_slice(&vec);
		assert_eq!(raw.as_slice(vec.len()), &vec);
		raw.deallocate(vec.len());
		Ok(())
	});
}

#[test]
fn test_from_slice_0() {
	from_slice::<0>();
}
#[test]
fn test_from_slice_8() {
	from_slice::<8>();
}
#[test]
fn test_from_slice_16() {
	from_slice::<16>();
}

// TODO: simulation testing
