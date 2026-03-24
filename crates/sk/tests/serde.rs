#[cfg(feature = "serde")]
mod test {
	use serde_test::{Token, assert_tokens};
	use sk::skbuf;

	#[test]
	fn test_skbuf_serialization() {
		let mut v = skbuf![1, 2, 3];

		assert_tokens(
			&v,
			&[
				Token::Struct {
					name: "SkBuf",
					len: 2,
				},
				Token::Str("items"),
				Token::Seq { len: Some(6) },
				Token::I32(1),
				Token::I32(1),
				Token::I32(2),
				Token::I32(2),
				Token::I32(3),
				Token::I32(3),
				Token::SeqEnd,
				Token::Str("edits"),
				Token::NewtypeStruct { name: "EditBuf" },
				Token::Seq { len: Some(3) },
				Token::U8(0),
				Token::U8(0),
				Token::U8(0),
				Token::SeqEnd,
				Token::StructEnd,
			],
		);

		v.set(0, 10);
		assert_tokens(
			&v,
			&[
				Token::Struct {
					name: "SkBuf",
					len: 2,
				},
				Token::Str("items"),
				Token::Seq { len: Some(6) },
				Token::I32(1),
				Token::I32(10),
				Token::I32(2),
				Token::I32(2),
				Token::I32(3),
				Token::I32(3),
				Token::SeqEnd,
				Token::Str("edits"),
				Token::NewtypeStruct { name: "EditBuf" },
				Token::Seq { len: Some(3) },
				Token::U8(3),
				Token::U8(0),
				Token::U8(0),
				Token::SeqEnd,
				Token::StructEnd,
			],
		);

		v.accept();
		assert_tokens(
			&v,
			&[
				Token::Struct {
					name: "SkBuf",
					len: 2,
				},
				Token::Str("items"),
				Token::Seq { len: Some(6) },
				Token::I32(1),
				Token::I32(10),
				Token::I32(2),
				Token::I32(2),
				Token::I32(3),
				Token::I32(3),
				Token::SeqEnd,
				Token::Str("edits"),
				Token::NewtypeStruct { name: "EditBuf" },
				Token::Seq { len: Some(3) },
				Token::U8(1),
				Token::U8(0),
				Token::U8(0),
				Token::SeqEnd,
				Token::StructEnd,
			],
		);
	}
}
