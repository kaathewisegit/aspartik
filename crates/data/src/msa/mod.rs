use anyhow::{Result, ensure};

use std::{
	collections::HashMap,
	hash::{DefaultHasher, Hasher},
	sync::Arc,
};

use crate::{
	DnaNucleotide,
	fasta::Record,
	seq::{Character, Sequence, SequenceMut},
};

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
		self.num_sequences + self.num_sites()
	}

	pub fn sequence_name(&self, index: usize) -> &str {
		&self.names[index]
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

	fn site_hash(&self, site: usize) -> u64 {
		let mut hasher = DefaultHasher::new();
		for i in (site..self.num_characters()).skip(self.num_sequences)
		{
			hasher.write_u8(self.data[i].to_byte());
		}

		hasher.finish()
	}

	fn is_same_site(&self, a: usize, b: usize) -> bool {
		for i in 0..self.num_sequences {
			if self.sequence(i)[a] != self.sequence(i)[b] {
				return false;
			}
		}

		true
	}

	pub fn deduplicate(&mut self) {
		// TODO: a bunch of nested vector allocations.  In general, feel
		// that there should be a more elegant way to do that.
		let mut map = HashMap::<u64, Vec<usize>>::new();

		for site in self.sites_iter() {
			let hash = self.site_hash(site);

			let Some(other_sites) = map.get_mut(&hash) else {
				// the hash is unique, meaning this site is also
				// unique
				map.insert(hash, vec![site]);
				continue;
			};

			for other_site in other_sites.iter() {
				if self.is_same_site(site, *other_site) {
					// the sites are the same, skip current one
					continue;
				}
			}

			// the sites were different after all
			other_sites.push(site);
		}

		let mut new_sites: Vec<usize> =
			map.values().flat_map(|v| v.iter().copied()).collect();
		// all indices are unique
		new_sites.sort_unstable();

		self.sites = Some(new_sites);
	}
}

impl Msa<DnaNucleotide> {
	pub fn base_frequencies(&self) -> [f64; 4] {
		let mut counts = [0usize; 4];
		let num_chars: usize = 0;

		for ch in self.data.as_ref() {
			match ch {
				DnaNucleotide::Adenine => counts[0] += 1,
				DnaNucleotide::Cytosine => counts[1] += 1,
				DnaNucleotide::Guanine => counts[2] += 1,
				DnaNucleotide::Thymine => counts[3] += 1,

				_ => continue,
			}
		}

		counts.map(|count| count as f64 / num_chars as f64)
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

		Some(out)
	}
}
