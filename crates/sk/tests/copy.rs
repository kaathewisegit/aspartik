use sk::SkBuf;

#[test]
fn basic() {
	let mut v = SkBuf::<i32>::from([1, 2, 3]);

	assert_eq!([1, 2, 3], v);

	v.set(0, 10);
	assert_eq!([10, 2, 3], v);

	v.set(2, 30);
	assert_eq!([10, 2, 30], v);

	v.accept();
	assert_eq!([10, 2, 30], v);

	v.set(1, 20);
	assert_eq!([10, 20, 30], v);

	v.reject();
	assert_eq!([10, 2, 30], v);
}
