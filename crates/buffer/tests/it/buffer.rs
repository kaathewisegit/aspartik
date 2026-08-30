use std::{hint::black_box, mem::drop};

use buffer::Buffer;

#[test]
fn init_drop() {
	drop(black_box(Buffer::<u8>::uninit(10)));
	drop(black_box(Buffer::<u8>::zeroed(10)));
}

#[test]
fn custom_alignment() {
	drop(black_box(Buffer::<u8, usize, 2>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 4>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 8>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 16>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 32>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 64>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 128>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 256>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 512>::uninit(10)));
	drop(black_box(Buffer::<u8, usize, 1024>::uninit(10)));
}

#[test]
fn zeroed() {
	assert_eq!(&Buffer::<u8>::zeroed(10)[..], &[0; 10]);
	assert_eq!(&Buffer::<i32>::zeroed(10)[..], &[0; 10]);
	assert_eq!(&Buffer::<u128>::zeroed(10)[..], &[0; 10]);
	assert_eq!(&Buffer::<f32>::zeroed(10)[..], &[0.0; 10]);
	assert_eq!(&Buffer::<f64>::zeroed(10)[..], &[0.0; 10]);
}
