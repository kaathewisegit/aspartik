//! Kitchen sink utilities.

use anyhow::{Result, bail};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PySliceIndices, PyTuple};

use data::{DnaNucleotide, Msa};

pub fn msa_to_likelihoods(msa: Msa<DnaNucleotide>) -> Vec<[f64; 4]> {
	let mut out = Vec::with_capacity(msa.num_sequences() * msa.num_sites());

	for seq in 0..msa.num_sequences() {
		for site in msa.sites_iter() {
			let char = msa.sequence(seq)[site];
			out.push(char.base_frequencies_denormalized())
		}
	}

	out
}

#[derive(Debug)]
pub struct SlicesIter {
	slices: Vec<PySliceIndices>,
	slice_index: usize,
	curr_index: isize,
}

impl Iterator for SlicesIter {
	type Item = usize;

	fn next(&mut self) -> Option<usize> {
		// get currently active slice
		let mut slice = self.slices.get(self.slice_index)?;

		self.curr_index += slice.step;
		// if we have overrun the current slice, advance to the
		// next one
		if self.curr_index >= slice.stop {
			self.slice_index += 1;
			slice = self.slices.get(self.slice_index)?;
			// set the index to the start of the next slice
			self.curr_index = slice.start;
		}

		Some(self.curr_index as usize)
	}
}

/// Iterator over the numbers specified by either a single slice or a tuple of
/// slices.
pub fn slices_iter(key: Bound<PyAny>, length: usize) -> Result<SlicesIter> {
	let mut slices = Vec::new();
	let length = length as isize;

	if let Ok(slice) = key.cast::<PySlice>() {
		let slice = slice.indices(length)?;
		slices.push(slice);
	} else if let Ok(tuple) = key.cast::<PyTuple>() {
		for item in tuple.into_iter() {
			let Ok(slice) = item.cast::<PySlice>() else {
				bail!(
					"Expected tuple members to be slices, got {}",
					item.get_type().name()?
				)
			};
			let slice = slice.indices(length)?;
			if slice.step < 0 {
				bail!("Negative slice step is not supported");
			}
			slices.push(slice);
		}

		// Slices will never be empty, because `list[]` is not a valid
		// Python syntax
	} else {
		bail!(
			"Expected a slice or a tuple of slices, got {}",
			key.get_type().name()?
		);
	}

	// Since `curr_index` is isize and the `indices` method will only return
	// positive values, we can do this
	let start = slices[0].start - slices[0].step;

	Ok(SlicesIter {
		slices,
		slice_index: 0,
		curr_index: start,
	})
}
