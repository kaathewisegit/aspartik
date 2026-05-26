use buffer::SliceBuffer;

#[test]
fn init_drop() {
	let buf: SliceBuffer<u8> = SliceBuffer::new(4, 10);
	assert_eq!(buf.len(), 10);
}

#[test]
fn len_matches_constructor() {
	let buf: SliceBuffer<f64> = SliceBuffer::new(3, 7);
	assert_eq!(buf.len(), 7);
}

#[test]
fn len_one() {
	let buf: SliceBuffer<u32> = SliceBuffer::new(5, 1);
	assert_eq!(buf.len(), 1);
}

#[test]
fn index_reads_zeroed() {
	let buf: SliceBuffer<u32> = SliceBuffer::new(3, 4);
	for i in 0..buf.len() {
		assert!(buf[i].iter().all(|&v| v == 0));
	}
}

#[test]
fn index_mut_write_read() {
	let mut buf: SliceBuffer<u8> = SliceBuffer::new(3, 4);
	buf[0].copy_from_slice(&[1, 2, 3]);
	buf[1].copy_from_slice(&[4, 5, 6]);
	buf[2].copy_from_slice(&[7, 8, 9]);
	buf[3].copy_from_slice(&[10, 11, 12]);

	assert_eq!(buf[0], [1, 2, 3]);
	assert_eq!(buf[1], [4, 5, 6]);
	assert_eq!(buf[2], [7, 8, 9]);
	assert_eq!(buf[3], [10, 11, 12]);
}

#[test]
fn get_unchecked_reads_zeroed() {
	let buf: SliceBuffer<u64> = SliceBuffer::new(2, 5);
	for i in 0..buf.len() {
		// SAFETY: `i < buf.len()`
		let slice = unsafe { buf.get_unchecked(i) };
		assert!(slice.iter().all(|&v| v == 0));
	}
}

#[test]
fn get_mut_unchecked_write_read() {
	let mut buf: SliceBuffer<u8> = SliceBuffer::new(2, 3);
	// SAFETY: `0, 1, 2 < buf.len()`
	unsafe {
		buf.get_mut_unchecked(0).copy_from_slice(&[10, 20]);
		buf.get_mut_unchecked(1).copy_from_slice(&[30, 40]);
		buf.get_mut_unchecked(2).copy_from_slice(&[50, 60]);
	}
	assert_eq!(buf[0], [10, 20]);
	assert_eq!(buf[1], [30, 40]);
	assert_eq!(buf[2], [50, 60]);
}

#[test]
fn custom_alignment() {
	let buf: SliceBuffer<u8, 64> = SliceBuffer::new(4, 8);
	assert_eq!(buf.len(), 8);
	let ptr = buf[0].as_ptr();
	assert_eq!(ptr.align_offset(64), 0);
}
