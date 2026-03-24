use linalg::Vector;

#[test]
fn from_element() {
	let vector: [i32; 4] = Vector::from_element(1i32);

	assert_eq!(vector, [1, 1, 1, 1]);
}

// TODO: numerical methods

// TODO: indexing

// TODO: arithmetic (arbtest, perfect precision)

// TODO: arithmetic/methods (arbtest, approximate)
