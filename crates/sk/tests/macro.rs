use sk::skbuf;

#[test]
fn list() {
	let v = skbuf![1, 2, 3];
	assert_eq!([1, 2, 3], v);

	let v = skbuf![Box::new(1), Box::new(2), Box::new(3)];
	assert_eq!([Box::new(1), Box::new(2), Box::new(3)], v);
}

#[test]
fn repeat() {
	let v = skbuf![1; 10];
	assert_eq!([1; 10], v);

	let v = skbuf![vec![20]; 20];
	assert_eq!(vec![vec![20]; 20], v);
}
