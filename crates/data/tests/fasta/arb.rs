use anyhow::Result;
use arbitrary::{Result as ArbResult, Unstructured};
use arbtest::arbtest;

use data::{
	DnaNucleotide,
	fasta::{FastaParser, Record},
};

fn arb_record(u: &mut Unstructured) -> ArbResult<Record<DnaNucleotide>> {
	let mut description = u.arbitrary::<String>()?;
	if let Some(index) = description.find(['\r', '\n']) {
		description.truncate(index);
	}

	let seq = u.arbitrary::<Vec<DnaNucleotide>>()?;

	Ok(Record::new(format!(">{description}"), seq))
}

fn parse_one_record(fasta: &str) -> Result<Record<DnaNucleotide>> {
	let mut parser = FastaParser::<DnaNucleotide>::new();
	for line in fasta.lines() {
		parser.parse_line(|_| Ok(()), line)?;
	}
	Ok(Record::new(parser.description.clone(), parser.seq.clone()))
}

#[test]
fn parse_record() {
	arbtest(|u| {
		let record = arb_record(u)?;
		let str_record = record.to_string();
		let parsed_record = parse_one_record(&str_record).unwrap();

		assert_eq!(record, parsed_record);

		Ok(())
	});
}
