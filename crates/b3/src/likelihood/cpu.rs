use anyhow::Result;
use num_traits::Inv;

use std::{
	ops::Range,
	slice::{from_raw_parts, from_raw_parts_mut},
};

use super::Calculator;

pub struct CpuLikelihood<const N: usize, F> {
	num_patterns: usize,
	num_leaves: usize,

	selectors: Vec<u8>,
	selectors_backup: Vec<u8>,

	/// Have the length of `num_patterns`
	samples: Vec<u8>,
	projections: Vec<[F; N]>,

	likelihoods: Vec<f64>,

	scales: Vec<bool>,
	scale_sums: Vec<u32>,
	scale_sums_backup: Vec<u32>,

	/// Scaling threshold on logarithmic scale
	scale_ln: u32,
	/// `e^(-scale_ln)`
	scale_threshold: F,
	/// `e^scale_ln`
	scale_mult: F,
}

impl Calculator<4, f64> for CpuLikelihood<4, f64> {
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[(usize, usize)],
		transitions: &[[[f64; 4]; 4]],
		leaves_end: usize,
		frequencies: [f64; 4],
	) -> Result<()> {
		let c_nodes: Vec<u32> =
			nodes.iter().map(|&n| n as u32).collect();
		let c_children: Vec<_> = children
			.iter()
			.map(|&(l, r)| [l as u32, r as u32])
			.collect();

		// SAFETY: TODO
		unsafe {
			ccalc::propose(
				c_nodes.as_ptr(),
				c_nodes.len() as u32,
				c_children.as_ptr() as *const u32,
				transitions.as_ptr() as *mut f64,
				leaves_end as u32,
				frequencies.into(),
				//
				self.num_patterns as u32,
				self.samples.as_ptr(),
				self.projections.as_mut_ptr() as *mut f64,
				self.selectors.as_mut_ptr(),
				self.likelihoods.as_mut_ptr(),
			)
		}

		Ok(())
	}

	fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()> {
		patterns.copy_from_slice(&self.likelihoods);

		for (i, scale_sum) in self.scale_sums.iter().enumerate() {
			patterns[i] -= f64::from(*scale_sum);
		}

		Ok(())
	}

	fn accept(&mut self) -> Result<()> {
		self.selectors_backup.copy_from_slice(&self.selectors);
		self.scale_sums_backup.copy_from_slice(&self.scale_sums);

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.selectors.copy_from_slice(&self.selectors_backup);
		self.scale_sums.copy_from_slice(&self.scale_sums_backup);

		Ok(())
	}
}

impl CpuLikelihood<4, f64> {
	pub fn new(
		num_patterns: usize,
		leaves: Vec<u8>,
		scale_ln: u32,
	) -> Self {
		let num_leaves = leaves.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let projections =
			vec![Default::default(); num_nodes * num_patterns * 2];
		let scales = vec![false; num_nodes * num_patterns * 2];
		let scale_sums = vec![0; num_patterns];

		let scale_ln_f64: f64 = scale_ln.into();
		let scale_mult = scale_ln_f64.exp();
		let scale_threshold = (-scale_ln_f64).exp();

		Self {
			num_patterns,
			num_leaves,

			selectors: vec![0; num_nodes],
			selectors_backup: vec![0; num_nodes],

			samples: leaves,
			projections,

			likelihoods: vec![f64::NAN; num_patterns],

			scales,
			scale_sums_backup: scale_sums.clone(),
			scale_sums,

			scale_ln,
			scale_threshold,
			scale_mult,
		}
	}
}
