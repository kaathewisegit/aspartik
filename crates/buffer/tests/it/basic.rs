use std::hint::black_box;

use buffer::Buffer;

#[test]
fn init_drop() {
	let _ = black_box(Buffer::<u8>::new(10));
}

#[test]
fn custom_alignment() {
	let _ = black_box(Buffer::<u8, usize, 2>::new(10));
	let _ = black_box(Buffer::<u8, usize, 4>::new(10));
	let _ = black_box(Buffer::<u8, usize, 8>::new(10));
	let _ = black_box(Buffer::<u8, usize, 16>::new(10));
	let _ = black_box(Buffer::<u8, usize, 32>::new(10));
	let _ = black_box(Buffer::<u8, usize, 64>::new(10));
	let _ = black_box(Buffer::<u8, usize, 128>::new(10));
	let _ = black_box(Buffer::<u8, usize, 256>::new(10));
	let _ = black_box(Buffer::<u8, usize, 512>::new(10));
	let _ = black_box(Buffer::<u8, usize, 1024>::new(10));
}
