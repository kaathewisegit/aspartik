use linalg::Vector;

#[test]
fn from_slice() {
	let slice = [0.1, 0.2, 0.3, 0.4];
	let vector = Vector::from(slice);

	assert_eq!(vector, slice);
}

#[test]
fn from_element() {
	let vector = Vector::from_element(1i32);

	assert_eq!(vector, [1, 1, 1, 1]);
}

#[test]
fn default() {
	let vector = Vector::<u8, 4>::default();

	assert_eq!(vector, [0u8, 0, 0, 0]);
}

// Mathematical constructors
// -------------------------
// TODO: all primitive types

#[test]
fn zeros() {
	assert_eq!(Vector::<u8, 4>::zeros(), [0u8, 0, 0, 0]);
}

#[test]
fn ones() {
	assert_eq!(Vector::<u8, 4>::ones(), [1u8, 1, 1, 1]);
}

#[test]
fn sbv() {
	const N: usize = 100;
	for i in 0..N {
		let mut slice = [0; N];
		slice[i] = 1;
		assert_eq!(Vector::<u8, N>::sbv(i), slice);
	}
}

// TODO: indexing

// TODO: arithmetic (arbtest, perfect precision)

// TODO: arithmetic/methods (arbtest, approximate)

#[test]
fn magnitude() {
	let v = Vector::from([3.0, 4.0, 12.0]);
	assert_eq!(v.magnitude(), 13.0);
}

#[test]
fn cosine_similarity() {
	let a = Vector::from([9.0, 3.0, 1.0]);
	let b = Vector::from([0.0, 1.0, 2.0]);
	assert_eq!(a.consine_similarity(&b), 0.2344036154692477);
}
