use data::{
	DnaNucleotide::{self, *},
	Msa,
	fasta::Record,
};

fn record(description: &str, seq: &[DnaNucleotide]) -> Record<DnaNucleotide> {
	Record::new(description.to_owned(), seq.to_vec())
}

#[test]
fn from_records() {
	let records = [
		record(">a", &[Adenine, Cytosine]),
		record(">b", &[Guanine, Thymine]),
	];

	let msa = Msa::from_records(records).unwrap();

	assert_eq!(msa.num_sequences(), 2);
	assert_eq!(msa.num_sites(), 2);
	assert_eq!(msa.sequence_names().get(0), "a");
	assert_eq!(msa.sequence_names().get(1), "b");
	assert_eq!(msa.sequence(0), &[Adenine, Cytosine]);
	assert_eq!(msa.sequence(1), &[Guanine, Thymine]);
}

#[test]
fn from_records_uneven_lengths() {
	let records =
		[record(">a", &[Adenine, Cytosine]), record(">b", &[Guanine])];

	Msa::from_records(records).unwrap_err();
}
