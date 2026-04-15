#![expect(clippy::undocumented_unsafe_blocks)]

use std::{
	alloc::{Layout, alloc},
	mem::drop,
};

use sk::{EditBuf, SkBuf};

#[test]
fn editbuf_basic() {
	let buffer = vec![0; 10];
	let (ptr, len, _) = buffer.into_raw_parts();

	let mut editbuf = unsafe { EditBuf::from_raw_parts(ptr, len) };
	editbuf.set_edited(5);
	editbuf.set_edited(1);
	editbuf.accept();
	editbuf.set_edited(9);
	editbuf.reject();

	drop(editbuf);
}

#[test]
fn skbuf_basic() {
	let len = 10;

	let buffer = vec![0; len];
	let (editbuf_ptr, _, _) = buffer.into_raw_parts();

	let buffer = vec![0.0f64; len * 2];
	let (ptr, _, _) = buffer.into_raw_parts();
	let mut sk = unsafe { SkBuf::from_raw_parts(ptr, editbuf_ptr, len) };

	sk.set(0, 10.0);
	sk.set(9, 1.0);
	sk.accept();
	sk.set(5, -1.0);
	sk.reject();

	drop(sk);
}

#[test]
fn skbuf_alloc() {
	let len = 10;
	let u8_layout = Layout::array::<u8>(len).unwrap();
	let f64_layout = Layout::array::<f64>(len * 2).unwrap();

	let mut sk = unsafe {
		let editbuf_ptr = alloc(u8_layout);
		std::ptr::write_bytes(editbuf_ptr, 0, len);

		let ptr = alloc(f64_layout) as *mut f64;

		SkBuf::from_raw_parts(ptr, editbuf_ptr, len)
	};

	sk.set(0, 10.0);
	sk.set(9, 1.0);
	sk.accept();
	sk.set(5, -1.0);
	sk.reject();

	drop(sk);
}
