use std::{
	collections::HashMap,
	hash::{DefaultHasher, Hasher},
	ops::{Index, IndexMut},
	sync::Arc,
};

use crate::{DnaNucleotide, seq::Character};

pub struct Msa<C: Character> {
	num_sequences: usize,
	num_sites: usize,
	names: Vec<String>,
	data: [C],
}

impl<C: Character> Msa<C> {
	pub fn num_sequences(&self) -> usize {
		self.num_sequences
	}

	pub fn num_sites(&self) -> usize {
		self.num_sites
	}

	pub fn num_characters(&self) -> usize {
		self.num_sequences + self.num_sites
	}

	pub fn sequence_name(&self, index: usize) -> &str {
		&self.names[index]
	}

	fn site_hash(&self, site: usize) -> u64 {
		let mut hasher = DefaultHasher::new();
		for i in (site..self.num_characters()).skip(self.num_sequences)
		{
			hasher.write_u8(self.data[i].into());
		}

		hasher.finish()
	}

	fn is_same_site(&self, a: usize, b: usize) -> bool {
		for i in 0..self.num_sequences {
			if self[i][a] != self[i][b] {
				return false;
			}
		}

		true
	}
}

impl Msa<DnaNucleotide> {
	pub fn base_frequencies(&self) -> [f64; 4] {
		let mut counts = [0usize; 4];
		let num_chars: usize = 0;

		for ch in &self.data {
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

impl<C: Character> Index<usize> for Msa<C> {
	type Output = [C];

	fn index(&self, index: usize) -> &[C] {
		let start = index * self.num_sites;
		let end = start + self.num_sites;
		&self.data[start..end]
	}
}

impl<C: Character> IndexMut<usize> for Msa<C> {
	fn index_mut(&mut self, index: usize) -> &mut [C] {
		let start = index * self.num_sites;
		let end = start + self.num_sites;
		&mut self.data[start..end]
	}
}

pub struct MsaView<C: Character> {
	msa: Arc<Msa<C>>,
	sites: Vec<usize>,
}

impl<C: Character> MsaView<C> {
	pub fn new<I>(msa: Arc<Msa<C>>, sites: I) -> Self
	where
		I: Iterator<Item = usize>,
	{
		MsaView {
			msa: msa.clone(),
			sites: sites.collect(),
		}
	}

	pub fn num_sites(&self) -> usize {
		self.sites.len()
	}

	pub fn deduplicate(&mut self) {
		// TODO: a bunch of nested vector allocations.  In general, feel
		// that there should be a more elegant way to do that.
		let mut map = HashMap::<u64, Vec<usize>>::new();

		for site in self.sites.iter().copied() {
			let hash = self.msa.site_hash(site);

			let Some(other_sites) = map.get_mut(&hash) else {
				// the hash is unique, meaning this site is also
				// unique
				map.insert(hash, vec![site]);
				continue;
			};

			for other_site in other_sites.iter() {
				if self.msa.is_same_site(site, *other_site) {
					// the sites are the same, skip current one
					continue;
				}
			}

			// the sites were different after all
			other_sites.push(site);
		}

		let new_sites: Vec<usize> =
			map.values().flat_map(|v| v.iter().copied()).collect();

		self.sites = new_sites;
		// all indices are unique
		self.sites.sort_unstable();
	}
}
