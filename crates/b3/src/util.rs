//! Kitchen sink utilities.
use anyhow::{bail, Result};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PySliceIndices, PyTuple};

use std::cmp::Ordering;

use crate::likelihood::Row;
use data::{seq::DnaSeq, DnaNucleotide};
use linalg::Vector;

fn compare_seqs(a: &[DnaNucleotide], b: &[DnaNucleotide]) -> Ordering {
	for (a, b) in a.iter().zip(b.iter()) {
		if a != b {
			let a_num: u8 = (*a).into();
			let b_num: u8 = (*b).into();
			return a_num.cmp(&b_num);
		}
	}
	Ordering::Equal
}

pub fn dna_to_rows(seqs: &[DnaSeq]) -> Vec<Vec<Row<4>>> {
	let seq_len = seqs[0].len();
	let num_seq = seqs.len();

	let mut transposed = vec![vec![]; seq_len];

	#[expect(clippy::needless_range_loop)]
	for nuc_idx in 0..seq_len {
		for seq_idx in 0..num_seq {
			transposed[nuc_idx].push(seqs[seq_idx][nuc_idx]);
		}
	}

	// Deduplicate same rows
	transposed.sort_by(|a, b| compare_seqs(a, b));
	transposed.dedup_by(|a, b| compare_seqs(a, b) == Ordering::Equal);

	let mut out = vec![
		vec![Vector::default(); transposed[0].len()];
		transposed.len()
	];

	// TODO: find a place for this
	fn to_row(base: &DnaNucleotide) -> Vector<f64, 4> {
		match base {
			DnaNucleotide::Adenine => [1.0, 0.0, 0.0, 0.0],
			DnaNucleotide::Cytosine => [0.0, 1.0, 0.0, 0.0],
			DnaNucleotide::Guanine => [0.0, 0.0, 1.0, 0.0],
			DnaNucleotide::Thymine => [0.0, 0.0, 0.0, 1.0],

			_ => [0.25, 0.25, 0.25, 0.25],
		}
		.into()
	}

	for i in 0..transposed.len() {
		for j in 0..transposed[0].len() {
			out[i][j] = to_row(&transposed[i][j])
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

	if let Ok(slice) = key.downcast::<PySlice>() {
		let slice = slice.indices(length)?;
		slices.push(slice);
	} else if let Ok(tuple) = key.downcast::<PyTuple>() {
		for item in tuple.into_iter() {
			let Ok(slice) = item.downcast::<PySlice>() else {
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
