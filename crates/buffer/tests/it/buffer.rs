use arbtest::arbtest;

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

#[test]
fn reallocate() {
	let mut buf = Buffer::<u8>::zeroed(4);
	for i in 0..4 {
		buf[i] = i as u8;
	}
	assert_eq!(&buf[..], &[0, 1, 2, 3]);

	buf.reallocate(6);
	for i in 4..6 {
		buf[i] = (i * 10) as u8;
	}
	assert_eq!(&buf[..], &[0, 1, 2, 3, 40, 50]);
	for i in 0..6 {
		buf[i] = (i * 10) as u8;
	}
	assert_eq!(&buf[..], &[0, 10, 20, 30, 40, 50]);

	buf.reallocate(2);
	assert_eq!(&buf[..], &[0, 10]);
}

fn from_slice<const A: usize>() {
	arbtest(|u| {
		let vec = u.arbitrary::<Vec<i32>>()?;
		let buf = Buffer::<i32, u32, A>::from_slice(&vec);
		assert_eq!(&*buf, &vec);
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
