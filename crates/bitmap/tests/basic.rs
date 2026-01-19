use bitmap::Bitmap;

#[test]
fn basic() {
	let mut b = Bitmap::new(10);
	b.set(1, true);
	b.set(2, true);
	b.set(9, true);

	assert!(b.at(1));
	assert!(b.at(2));
	assert!(b.at(9));
	assert!(!b.at(0));
	assert!(!b.at(3));
	assert!(!b.at(4));

	b.set(9, false);
	assert!(!b.at(9))
}
