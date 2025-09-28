use data::DnaNucleotide::{self, *};
use data::seq::*;

#[test]
fn parse_err() {
	parse_str::<DnaNucleotide>("ACTGxACTG").unwrap_err();
}

#[test]
fn count() {
	let s = dna![
		"AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC"
	];

	assert_eq!(s.count(Adenine), 20);
	assert_eq!(s.count(Cytosine), 12);
	assert_eq!(s.count(Guanine), 17);
	assert_eq!(s.count(Thymine), 21);
}

#[test]
fn dna_complement() {
	let s = dna!["AAAACCCGGT"];

	assert_eq!(s.reverse_complement().to_string(), "ACCGGGTTTT");
}

#[test]
fn hamming() {
	let s1 = dna!["GAGCCTACTAACGGGAT"];
	let s2 = dna!["CATCGTAATGACGGCCT"];

	assert_eq!(distance::hamming(&s1, &s2).unwrap(), 7);
}

#[test]
fn index() {
	let mut s = parse_str("ACGT").unwrap();
	assert_eq!(s[0], Adenine);
	assert_eq!(s[1], Cytosine);
	assert_eq!(s[2], Guanine);
	assert_eq!(s[3], Thymine);

	s[0] = Thymine;
	s[1] = Cytosine;
	s[2] = Guanine;
	s[3] = Adenine;
	assert_eq!(s[0], Thymine);
	assert_eq!(s[1], Cytosine);
	assert_eq!(s[2], Guanine);
	assert_eq!(s[3], Adenine);
}
