use anyhow::{Result, ensure};

use std::{
	cmp::Ordering,
	ops::{Bound, RangeBounds},
	sync::Arc,
};

use crate::{
	DnaNucleotide,
	fasta::Record,
	seq::{Character, Sequence, SequenceMut},
};
use linalg::Vector;

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone)]
pub struct Msa<C: Character> {
	num_sequences: usize,
	num_sites_total: usize,
	sites: Option<Vec<usize>>,
	names: Arc<[String]>,
	data: Sequence<C>,
}

impl<C: Character> Msa<C> {
	pub fn new(
		num_sequences: usize,
		num_sites: usize,
		names: Arc<[String]>,
		data: Sequence<C>,
	) -> Result<Self> {
		ensure!(num_sequences * num_sites == data.len());
		ensure!(names.len() == num_sequences);

		Ok(Self {
			num_sequences,
			num_sites_total: num_sites,
			sites: None,
			names,
			data,
		})
	}

	pub fn from_fasta<I, R>(records: I) -> Result<Self>
	where
		I: IntoIterator<Item = R>,
		R: AsRef<Record<C>>,
	{
		let mut num_sites = 0;
		let mut num_sequences = 0;

		let mut data = SequenceMut::new();
		let mut names = Vec::new();

		for record in records.into_iter() {
			let record = record.as_ref();

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
			num_sites_total: num_sites,
			sites: None,
			names: names.into(),
			data: data.into(),
		})
	}

	pub fn num_sequences(&self) -> usize {
		self.num_sequences
	}

	pub fn num_sites(&self) -> usize {
		self.sites
			.as_ref()
			.map(|s| s.len())
			.unwrap_or(self.num_sites_total)
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

	pub fn sequence(&self, index: usize) -> Sequence<C> {
		let start = index * self.num_sites_total;
		let end = start + self.num_sites_total;
		self.data.slice(start..end)
	}

	pub fn sites_iter(&self) -> SitesIter<'_> {
		SitesIter {
			sites: self.sites.as_deref(),
			current: 0,
			end: self.num_sites(),
		}
	}

	pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
		let mut out = self.clone();

		let mut sites = match self.sites.as_ref() {
			Some(sites) => sites.clone(),
			None => (0..self.num_sites()).collect(),
		};

		let start = match range.start_bound() {
			Bound::Included(i) => *i + 1,
			Bound::Excluded(i) => *i,
			Bound::Unbounded => sites.len(),
		};
		let end = match range.end_bound() {
			Bound::Included(i) => *i + 1,
			Bound::Excluded(i) => *i,
			Bound::Unbounded => sites.len(),
		};
		let length = end - start;
		sites.copy_within(range, 0);
		sites.truncate(length);

		out.sites = Some(sites);
		out
	}

	fn compare_sites(&self, a: &usize, b: &usize) -> Ordering {
		for seq in 0..self.num_sequences {
			let seq = self.sequence(seq);
			let a = seq[*a];
			let b = seq[*b];

			if a != b {
				let (a_b, b_b) = (a.to_byte(), b.to_byte());
				return a_b.cmp(&b_b);
			}
		}
		Ordering::Equal
	}

	pub fn deduplicate(&self) -> Self {
		let mut sites: Vec<_> = self.sites_iter().collect();
		sites.sort_by(|a, b| self.compare_sites(a, b));
		sites.dedup_by(|a, b| self.compare_sites(a, b).is_eq());

		let mut out = self.clone();
		out.sites = Some(sites);
		out
	}
}

impl Msa<DnaNucleotide> {
	pub fn base_frequencies(&self) -> [f64; 4] {
		let mut counts = Vector::zeros();
		let num_chars = self.num_characters();

		for seq in 0..self.num_sequences {
			let seq = self.sequence(seq);

			for site in self.sites_iter() {
				counts += seq[site].base_frequencies();
			}
		}

		counts /= num_chars as f64;
		counts.into()
	}
}

pub struct SitesIter<'a> {
	sites: Option<&'a [usize]>,
	current: usize,
	end: usize,
}

impl<'a> Iterator for SitesIter<'a> {
	type Item = usize;

	fn next(&mut self) -> Option<usize> {
		if self.current == self.end {
			return None;
		}

		let out;
		if let Some(sites) = self.sites {
			out = sites[self.current];
		} else {
			out = self.current;
		}

		self.current += 1;

		Some(out)
	}
}
