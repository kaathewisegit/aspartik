use data::seq::*;
use data::DnaNucleotide::{self, *};

#[test]
fn decode() {
	parse_str::<Vec<DnaNucleotide>>("ACTGxACTG").unwrap_err();
}

#[test]
fn count() {
	let s = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGC";
	let s: Vec<DnaNucleotide> = parse_str(s).unwrap();

	assert_eq!(s.count(Adenine), 20);
	assert_eq!(s.count(Cytosine), 12);
	assert_eq!(s.count(Guanine), 17);
	assert_eq!(s.count(Thymine), 21);
}

#[test]
fn dna_complement() {
	let s: Vec<DnaNucleotide> = parse_str("AAAACCCGGT").unwrap();

	assert_eq!(s.reverse_complement().to_string(), "ACCGGGTTTT");
}

#[test]
fn hamming() {
	let s1: Vec<DnaNucleotide> = parse_str("GAGCCTACTAACGGGAT").unwrap();
	let s2: Vec<DnaNucleotide> = parse_str("CATCGTAATGACGGCCT").unwrap();

	assert_eq!(distance::hamming(s1, s2).unwrap(), 7);
}

#[test]
fn index() {
	let mut s: Vec<DnaNucleotide> = parse_str("ACGT").unwrap();
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

#[test]
fn iter() {
	let s: Vec<DnaNucleotide> = parse_str("GAGCCT").unwrap();
	let mut iter = s.iter().copied();
	assert_eq!(iter.next(), Some(Guanine));
	assert_eq!(iter.next(), Some(Adenine));
	assert_eq!(iter.next(), Some(Guanine));
	assert_eq!(iter.next(), Some(Cytosine));
	assert_eq!(iter.next(), Some(Cytosine));
	assert_eq!(iter.next(), Some(Thymine));
}
