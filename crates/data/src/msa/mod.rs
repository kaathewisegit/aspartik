use anyhow::{Result, ensure};
use rand::Rng;

use std::{cmp::Ordering, ops::Range};

use crate::{
	DnaNucleotide,
	fasta::Record,
	seq::{Character, Sequence, SequenceMut, SequenceRef},
};

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msa<C: Character> {
	num_sequences: usize,
	num_sites: usize,
	names: Box<[String]>,
	data: Sequence<C>,
}

impl<C: Character> Msa<C> {
	pub fn new(
		num_sequences: usize,
		num_sites: usize,
		names: Box<[String]>,
		data: Sequence<C>,
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

	pub fn from_fasta<I>(records: I) -> Result<Self>
	where
		I: IntoIterator<Item = Result<Record<C>>>,
	{
		let mut num_sites = 0;
		let mut num_sequences = 0;

		let mut data = SequenceMut::new();
		let mut names = Vec::new();

		for record in records.into_iter() {
			let record = record?;

			if num_sites == 0 {
				num_sites = record.sequence().len();
			}

			ensure!(num_sites == record.sequence().len());

			data.extend(record.sequence());
			names.push(record.id().to_owned());
			num_sequences += 1;
		}

		Ok(Self {
			num_sequences,
			num_sites,
			names: names.into(),
			data: data.into(),
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

	pub fn sequence(&self, index: usize) -> SequenceRef<'_, C> {
		let start = index * self.num_sites;
		let end = start + self.num_sites;
		self.data.slice(start..end)
	}

	pub fn sequence_owned(&self, index: usize) -> Sequence<C> {
		let start = index * self.num_sites;
		let end = start + self.num_sites;
		self.data.slice_owned(start..end)
	}

	pub fn sequences(&self) -> impl Iterator<Item = SequenceRef<'_, C>> {
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
		let seq = Sequence::random(num_sequences * num_sites, rng);
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
