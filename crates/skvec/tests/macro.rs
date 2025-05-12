use skvec::{skvec, SkVec};

#[test]
fn empty() {
	// Copy
	let v: SkVec<usize> = skvec![];
	assert!(v.is_empty());

	// Clone
	let v: SkVec<Box<usize>> = skvec![];
	assert!(v.is_empty());
}

#[test]
fn list() {
	let v = skvec![1, 2, 3];
	assert_eq!([1, 2, 3], v);

	let v = skvec![Box::new(1), Box::new(2), Box::new(3)];
	assert_eq!([Box::new(1), Box::new(2), Box::new(3)], v);
}

#[test]
fn repeat() {
	let v = skvec![1; 10];
	assert_eq!([1; 10], v);

	let v = skvec![vec![20]; 20];
	assert_eq!(vec![vec![20]; 20], v);
}
