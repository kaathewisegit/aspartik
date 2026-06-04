use b3::SkSliceBuf;

#[test]
fn new() {
	let buf: SkSliceBuf<f64> = SkSliceBuf::new(4, 3);
	assert_eq!(buf[0], [0.0; 4]);
	assert_eq!(buf[1], [0.0; 4]);
	assert_eq!(buf[2], [0.0; 4]);
}

#[test]
fn update_single() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(4, 3);
	buf.update(1).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
	assert_eq!(buf[0], [0.0; 4]);
	assert_eq!(buf[1], [1.0, 2.0, 3.0, 4.0]);
	assert_eq!(buf[2], [0.0; 4]);
}

#[test]
fn accept() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(2, 2);
	buf.update(0).copy_from_slice(&[1.0, 2.0]);
	buf.accept();
	assert_eq!(buf[0], [1.0, 2.0]);
	assert!(buf[1].iter().all(|&v| v == 0.0));
}

#[test]
fn reject() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(2, 2);
	buf.update(0).copy_from_slice(&[1.0, 2.0]);
	buf.reject();
	assert_eq!(buf[0], [0.0; 2]);
}

#[test]
fn accept_update() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(2, 1);

	buf.update(0).copy_from_slice(&[1.0, 2.0]);
	buf.accept();
	assert_eq!(buf[0], [1.0, 2.0]);

	buf.update(0).copy_from_slice(&[3.0, 4.0]);
	assert_eq!(buf[0], [3.0, 4.0]);

	buf.reject();
	assert_eq!(buf[0], [1.0, 2.0]);
}

#[test]
fn reject_update_accept() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(2, 1);

	buf.update(0).copy_from_slice(&[1.0, 2.0]);
	buf.reject();
	assert_eq!(buf[0], [0.0; 2]);

	buf.update(0).copy_from_slice(&[5.0, 6.0]);
	assert_eq!(buf[0], [5.0, 6.0]);

	buf.accept();
	assert_eq!(buf[0], [5.0, 6.0]);
}

#[test]
fn multi_index() {
	let mut buf: SkSliceBuf<u32> = SkSliceBuf::new(3, 3);
	buf.update(0).copy_from_slice(&[1, 2, 3]);
	buf.update(2).copy_from_slice(&[7, 8, 9]);

	assert_eq!(buf[0], [1, 2, 3]);
	assert_eq!(buf[1], [0, 0, 0]);
	assert_eq!(buf[2], [7, 8, 9]);

	buf.accept();
	assert_eq!(buf[0], [1, 2, 3]);
	assert_eq!(buf[1], [0, 0, 0]);
	assert_eq!(buf[2], [7, 8, 9]);

	buf.update(0).copy_from_slice(&[10, 11, 12]);
	buf.update(2).copy_from_slice(&[20, 21, 22]);
	buf.reject();

	assert_eq!(buf[0], [1, 2, 3]);
	assert_eq!(buf[1], [0, 0, 0]);
	assert_eq!(buf[2], [7, 8, 9]);
}

#[test]
fn multi_accept() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(1, 1);

	for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
		buf.update(0).copy_from_slice(&[v]);
		buf.accept();
		assert_eq!(buf[0], [v]);
	}
}

#[test]
fn reject_noop() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(2, 1);
	buf.reject();
	assert_eq!(buf[0], [0.0; 2]);
}

#[test]
fn accept_noop() {
	let mut buf: SkSliceBuf<f64> = SkSliceBuf::new(2, 1);
	buf.accept();
	assert_eq!(buf[0], [0.0; 2]);
}
