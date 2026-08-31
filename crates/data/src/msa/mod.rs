use anyhow::{Result, ensure};
use rand::Rng;

use std::{cmp::Ordering, io::BufRead, mem, ops::Range};

use crate::{
	DnaNucleotide,
	fasta::FastaParser,
	seq::{Character, random_dna},
};

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msa<C: Character> {
	num_sequences: usize,
	num_sites: usize,
	names: Box<[String]>,
	data: Vec<C>,
}

impl<C: Character> Msa<C> {
	pub fn new(
		num_sequences: usize,
		num_sites: usize,
		names: Box<[String]>,
		data: Vec<C>,
	) -> Result<Self> {
		ensure!(num_sequences * num_sites == data.len());
		ensure!(names.len() == num_sequences);

		Ok(Self {
			num_sequences,
			num_sites,
			names,
			data,
		})
	}

	pub fn from_fasta_reader<R>(reader: R) -> Result<Self>
	where
		R: BufRead,
	{
		let mut num_sites = 0;
		let mut num_sequences = 0;

		let mut data = Vec::new();
		let mut names = Vec::new();

		let mut add_record = |parser: &mut FastaParser<C>| {
			if num_sites == 0 {
				num_sites = parser.seq.len();
			}
			ensure!(num_sites == parser.seq.len());

			data.append(&mut parser.seq);
			names.push(mem::take(&mut parser.description));
			num_sequences += 1;
			Ok(())
		};

		let mut parser = FastaParser::new();

		// TODO: use read_string to avoid allocating strings
		for line in reader.lines() {
			parser.parse_line(&mut add_record, &line?)?;
		}

		add_record(&mut parser)?;

		Ok(Self {
			num_sequences,
			num_sites,
			names: names.into(),
			data,
		})
	}

	pub fn num_sequences(&self) -> usize {
		self.num_sequences
	}

	pub fn num_sites(&self) -> usize {
		self.num_sites
	}

	pub fn num_characters(&self) -> usize {
		self.num_sequences * self.num_sites()
	}

	pub fn sequence_name(&self, index: usize) -> &str {
		&self.names[index]
	}

	pub fn sequence_names(&self) -> &[String] {
		&self.names
	}

	pub fn sequence(&self, index: usize) -> &[C] {
		let start = index * self.num_sites;
		let end = start + self.num_sites;
		&self.data[start..end]
	}

	pub fn sequence_owned(&self, index: usize) -> Vec<C> {
		let start = index * self.num_sites;
		let end = start + self.num_sites;
		self.data[start..end].to_vec()
	}

	pub fn sequences(&self) -> impl Iterator<Item = &[C]> {
		(0..self.num_sequences()).map(|i| self.sequence(i))
	}

	pub fn sites(&self) -> Range<usize> {
		0..self.num_sites
	}

	pub fn compare_sites(&self, a: &usize, b: &usize) -> Ordering {
		for seq in 0..self.num_sequences {
			let seq = self.sequence_owned(seq);
			let a = seq[*a];
			let b = seq[*b];

			if a != b {
				let (a_b, b_b) = (a.into_byte(), b.into_byte());
				return a_b.cmp(&b_b);
			}
		}
		Ordering::Equal
	}
}

fn add_assign_arr(to: &mut [f64; 4], from: [f64; 4]) {
	for i in 0..4 {
		to[i] += from[i];
	}
}

impl Msa<DnaNucleotide> {
	pub fn random<R: Rng>(
		num_sequences: usize,
		num_sites: usize,
		names: Box<[String]>,
		rng: &mut R,
	) -> Result<Self> {
		let seq = random_dna(num_sequences * num_sites, rng);
		Self::new(num_sequences, num_sites, names, seq)
	}

	pub fn base_frequencies(&self) -> [f64; 4] {
		let mut counts = [0.0; 4];
		let num_chars = self.num_characters();

		for seq in 0..self.num_sequences {
			let seq = self.sequence_owned(seq);

			for char in seq.iter() {
				add_assign_arr(
					&mut counts,
					char.base_frequencies(),
				);
			}
		}

		for count in &mut counts {
			*count /= num_chars as f64;
		}
		counts
	}
}
