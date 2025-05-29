use anyhow::{ensure, Result};

use std::cmp::min;

use super::{Character, Seq};

pub fn hamming<S1, S2, C>(a: S1, b: S2) -> Result<usize>
where
	S1: Seq<Character = C>,
	S2: Seq<Character = C>,
	C: Character,
{
	ensure!(a.len() == b.len(), "Sequences have different lengths");

	let a = a.as_slice();
	let b = b.as_slice();

	let mut count = 0;

	for (a, b) in a.iter().zip(b.iter()) {
		if a != b {
			count += 1;
		}
	}

	Ok(count)
}

pub fn levenshtein<S1, S2, C>(a: S1, b: S2) -> Result<usize>
where
	S1: Seq<Character = C>,
	S2: Seq<Character = C>,
	C: Character,
{
	let a = a.as_slice();
	let b = b.as_slice();

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
