#[cfg(feature = "serde")]
mod test {
	use serde_json::to_string;
	use skvec::skvec;

	#[test]
	fn serialize() {
		let mut v = skvec![1, 2, 3];
		let json = to_string(&v).unwrap();
		assert_eq!(
			json,
			r#"{"inner":[1,1,2,2,3,3],"metadata":[0,0,0]}"#
		);

		v.set(0, 10);
		let json = to_string(&v).unwrap();
		assert_eq!(
			json,
			r#"{"inner":[1,10,2,2,3,3],"metadata":[3,0,0]}"#
		);

		v.accept();
		let json = to_string(&v).unwrap();
		assert_eq!(
			json,
			r#"{"inner":[1,10,2,2,3,3],"metadata":[1,0,0]}"#
		);
	}
}
