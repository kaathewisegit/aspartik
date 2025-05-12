use skvec::SkVec;

#[test]
fn basic() {
	let mut v = SkVec::<i32>::new();

	v.push(1);
	v.push(2);
	v.push(3);
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
