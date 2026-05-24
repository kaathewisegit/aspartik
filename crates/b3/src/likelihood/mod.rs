use anyhow::Result;
use pyo3::prelude::*;

use std::collections::HashMap;

use data::{DnaNucleotide, Msa, seq::Character};

mod compound;
mod dna;
mod gamma;
mod state;

pub use compound::CompoundLikelihood;
pub use dna::DnaLikelihood;
pub use gamma::GammaLikelihood;
pub use state::StateLikelihood;

#[derive(FromPyObject, IntoPyObject)]
pub enum Likelihood {
	Dna(Py<DnaLikelihood>),
	Gamma(Py<GammaLikelihood>),
	Compound(Py<CompoundLikelihood>),
}

impl Likelihood {
	pub fn likelihood(&self) -> Result<f64> {
		match self {
			Likelihood::Dna(l) => l.get().likelihood(),
			Likelihood::Gamma(l) => l.get().likelihood(),
			Likelihood::Compound(l) => l.get().likelihood(),
		}
	}

	pub fn accept(&self) -> Result<()> {
		match self {
			Likelihood::Dna(l) => l.get().accept(),
			Likelihood::Gamma(l) => l.get().accept(),
			Likelihood::Compound(l) => l.get().accept(),
		}
	}

	pub fn reject(&self) -> Result<()> {
		match self {
			Likelihood::Dna(l) => l.get().reject(),
			Likelihood::Gamma(l) => l.get().reject(),
			Likelihood::Compound(l) => l.get().reject(),
		}
	}

	pub fn clone_ref(&self, py: Python) -> Self {
		match self {
			Self::Dna(l) => Self::Dna(l.clone_ref(py)),
			Self::Gamma(l) => Self::Gamma(l.clone_ref(py)),
			Self::Compound(l) => Self::Compound(l.clone_ref(py)),
		}
	}
}

fn deduplicate(msa: &Msa<DnaNucleotide>) -> (Vec<u8>, Vec<u32>) {
	let mut hashes =
		Vec::<(usize, blake3::Hash)>::with_capacity(msa.num_sites());

	let mut scratch = Vec::<u8>::with_capacity(msa.num_sequences());
	let mut hasher = blake3::Hasher::new();
	for site in 0..msa.num_sites() {
		for seq in msa.sequences() {
			scratch.push(seq[site].into_byte());
		}
		hasher.update(&scratch);
		scratch.clear();

		hashes.push((site, hasher.finalize()));
		hasher.reset();
	}

	// hash -> (index, count)
	let mut map = HashMap::<blake3::Hash, (usize, u32)>::new();

	for (index, hash) in &hashes {
		if let Some((_, count)) = map.get_mut(hash) {
			// there's an earlier site with the same contents
			*count += 1;
		} else {
			map.insert(*hash, (*index, 1));
		}
	}

	let mut pairs: Vec<_> = map.values().collect();
	pairs.sort_by_key(|(index, _)| index);

	let (indices, weights): (Vec<_>, Vec<_>) =
		pairs.iter().copied().copied().unzip();

	let mut leaves =
		Vec::with_capacity(msa.num_sequences() * indices.len());

	fn char_to_u8(ch: DnaNucleotide) -> u8 {
		match ch {
			DnaNucleotide::Gap => 0b1111,
			ch => ch.into_byte(),
		}
	}

	for seq in msa.sequences() {
		for site in indices.iter().copied() {
			let char = seq[site];
			leaves.push(char_to_u8(char))
		}
	}

	(leaves, weights)
}
