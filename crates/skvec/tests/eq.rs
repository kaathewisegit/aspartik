use skvec::skvec;

#[test]
fn basic_eq() {
	let v = skvec![1, 2, 3];

	assert_eq!(v, vec![1, 2, 3]);
	assert_eq!(v, [1, 2, 3].as_slice());
	assert_eq!(v, [1, 2, 3]);

	assert_eq!(vec![1, 2, 3], v);
	assert_eq!([1, 2, 3].as_slice(), v);
	assert_eq!([1, 2, 3], v);
}

#[test]
fn basic_partial_eq() {
	let v = skvec![1.0, 2.0, 3.0];

	assert_eq!(v, vec![1.0, 2.0, 3.0]);
	assert_eq!(v, [1.0, 2.0, 3.0].as_slice());
	assert_eq!(v, [1.0, 2.0, 3.0]);

	assert_eq!(vec![1.0, 2.0, 3.0], v);
	assert_eq!([1.0, 2.0, 3.0].as_slice(), v);
	assert_eq!([1.0, 2.0, 3.0], v);
}

// TODO: run the vector through a few epochs and check the same thing
