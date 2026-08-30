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
