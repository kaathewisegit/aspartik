use anyhow::Result;
use arbitrary::{Result as ArbResult, Unstructured};
use arbtest::arbtest;

use data::{
	DnaNucleotide,
	fasta::{FastaParser, Record},
	seq::Sequence,
};

fn arb_record(u: &mut Unstructured) -> ArbResult<Record<DnaNucleotide>> {
	let mut description = u.arbitrary::<String>()?;
	if let Some(index) = description.find(['\r', '\n']) {
		description.truncate(index);
	}

	let seq = u.arbitrary::<Sequence<DnaNucleotide>>()?;

	Ok(Record::new(format!(">{description}"), seq))
}

fn parse_one_record(fasta: &str) -> Result<Record<DnaNucleotide>> {
	let mut parser = FastaParser::<DnaNucleotide>::new();
	let mut lines = fasta.lines();

	let mut out = None;

	loop {
		let line = lines.next();
		let result = parser.read_line(line)?;
		if let Some(record) = result {
			if out.is_none() {
				out = Some(record);
				continue;
			} else {
				panic!("second record")
			}
		}
		if line.is_none() {
			break;
		}
	}

	Ok(out.unwrap())
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
