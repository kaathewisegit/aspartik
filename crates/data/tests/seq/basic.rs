use data::DnaNucleotide::{self, *};
use data::seq::*;

#[test]
fn decode() {
	parse_str::<DnaNucleotide>("ACTGxACTG").unwrap_err();
}

#[test]
fn count() {
	let s = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC";
	let s = parse_str(s).unwrap().into_sequence();

	assert_eq!(s.count(Adenine), 20);
	assert_eq!(s.count(Cytosine), 12);
	assert_eq!(s.count(Guanine), 17);
	assert_eq!(s.count(Thymine), 21);
}

#[test]
fn dna_complement() {
	let s = parse_str("AAAACCCGGT").unwrap().into_sequence();

	assert_eq!(s.reverse_complement().to_string(), "ACCGGGTTTT");
}

#[test]
fn hamming() {
	let s1 = parse_str::<DnaNucleotide>("GAGCCTACTAACGGGAT")
		.unwrap()
		.into_sequence();
	let s2 = parse_str::<DnaNucleotide>("CATCGTAATGACGGCCT")
		.unwrap()
		.into_sequence();

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
