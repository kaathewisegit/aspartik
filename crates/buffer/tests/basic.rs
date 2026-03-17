use std::hint::black_box;

use buffer::Buffer;

#[test]
fn init_drop() {
	let _ = black_box(Buffer::<u8>::new(10));
}

#[test]
fn custom_alignment() {
	let _ = black_box(Buffer::<u8, 2>::new(10));
	let _ = black_box(Buffer::<u8, 4>::new(10));
	let _ = black_box(Buffer::<u8, 8>::new(10));
	let _ = black_box(Buffer::<u8, 16>::new(10));
	let _ = black_box(Buffer::<u8, 32>::new(10));
	let _ = black_box(Buffer::<u8, 64>::new(10));
	let _ = black_box(Buffer::<u8, 128>::new(10));
	let _ = black_box(Buffer::<u8, 256>::new(10));
	let _ = black_box(Buffer::<u8, 512>::new(10));
	let _ = black_box(Buffer::<u8, 1024>::new(10));
}
