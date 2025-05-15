#![allow(dead_code)]

use nom::{
	branch::alt,
	bytes::{tag, take_till},
	character::complete::{char, multispace0, none_of},
	combinator::value,
	multi::fold_many0,
	number::complete::float,
	sequence::delimited,
	IResult, Parser,
};

use super::Node;

macro_rules! ws {
	($f:expr) => {
		|s| delimited(multispace0, $f, multispace0).parse(s)
	};
}

fn bare(input: &str) -> IResult<&str, String> {
	let ch = none_of(" \t\n,;[\"");

	fold_many0(ch, String::new, |mut string, mut fragment| {
		if fragment == '_' {
			fragment = ' ';
		}
		string.push(fragment);
		string
	})
	.parse(input)
}

fn quoted(input: &str) -> IResult<&str, String> {
	let ch = alt((none_of("\"\\"), value('"', tag("\\\""))));

	let build = fold_many0(ch, String::new, |mut string, fragment| {
		string.push(fragment);
		string
	});

	delimited(char('"'), build, char('"')).parse(input)
}

fn name(input: &str) -> IResult<&str, String> {
	// Try `quoted` first
	alt((quoted, bare)).parse(input)
}

fn comment(input: &str) -> IResult<&str, String> {
	// XXX: can Newick contain nested brackets?
	delimited(char('['), take_till(|c| c != ']'), char(']'))
		.parse(input)
		.map(|(rest, s)| (rest, s.to_owned()))
}

fn distance(input: &str) -> IResult<&str, f64> {
	float(input).map(|(rest, f)| (rest, f as f64))
}

fn body(input: &str) -> IResult<&str, Node> {
	let (rest, name) = ws!(name)(input)?;
	let (rest, attributes) = ws!(comment)(rest)?;
	let (rest, distance) = ws!(distance)(rest)?;

	Ok((rest, Node::new(name, attributes, distance)))
}

#[cfg(test)]
mod test {
	use super::*;

	macro_rules! check {
		($tests:ident, $parser:expr) => {
			for (string, answer) in $tests {
				let result = $parser(string);
				assert!(result.is_ok());
				assert_eq!(
					(answer.0, answer.1.to_owned()),
					result.unwrap()
				);
			}
		};
	}

	#[test]
	fn test_name() {
		let bare_tests = [
			("bareIdent", ("", "bareIdent")),
			("rest of it", (" of it", "rest")),
			("under_score", ("", "under score")),
			("__and_trailing___", ("", "  and trailing   ")),
		];
		check!(bare_tests, bare);

		let quoted_tests = [
			(r#""a quoted string""#, ("", "a quoted string")),
			(
				r#""quoted with" trailing"#,
				(" trailing", "quoted with"),
			),
			(r#""esc\"ape""#, ("", "esc\"ape")),
		];
		check!(quoted_tests, quoted);

		check!(bare_tests, name);
		check!(quoted_tests, name);
	}
}
