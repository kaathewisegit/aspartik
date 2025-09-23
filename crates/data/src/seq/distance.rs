use anyhow::{Result, ensure};

use std::cmp::min;

use super::Character;

pub fn hamming<C: Character>(a: &[C], b: &[C]) -> Result<usize> {
	ensure!(a.len() == b.len(), "Sequences have different lengths");

	let mut count = 0;

	for (a, b) in a.iter().zip(b.iter()) {
		if a != b {
			count += 1;
		}
	}

	Ok(count)
}

pub fn levenshtein<C: Character>(a: &[C], b: &[C]) -> Result<usize> {
	let b_len = b.len();

	let mut cache: Vec<usize> = (1..b_len + 1).collect();

	let mut out = b_len;

	for (i, a_elem) in a.iter().enumerate() {
		out = i + 1;

		let mut distance_b = i;

		for (j, b_elem) in b.iter().enumerate() {
			let cost = usize::from(a_elem != b_elem);

			let distance_a = distance_b + cost;

			distance_b = cache[j];

			out = min(out + 1, min(distance_a, distance_b + 1));

			cache[j] = out;
		}
	}

	Ok(out)
}
