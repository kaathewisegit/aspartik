use anyhow::Result;

use super::Calculator;
use crate::{Transitions, parameters::Tree};
use buffer::Buffer;

#[allow(dead_code)]
pub struct Cpu4 {
	num_patterns: usize,

	pattern_weights: Vec<u32>,

	selectors: Vec<u8>,
	selectors_backup: Vec<u8>,

	/// Have the length of `num_patterns`
	samples: Vec<u8>,
	/// 32-bit alignment for aligned loads into YMM registers.
	projections: Buffer<[f64; 4], 32>,

	likelihoods: Vec<f64>,

	scales: Vec<bool>,
	scale_sums: Vec<u32>,
	scale_sums_backup: Vec<u32>,

	/// Scaling threshold on logarithmic scale
	scale_ln: u32,
	/// `e^(-scale_ln)`
	scale_threshold: f64,
	/// `e^scale_ln`
	scale_mult: f64,
}

impl Calculator<4, f64> for Cpu4 {
	fn likelihood(
		&mut self,
		tree: &Tree,
		transitions: &Transitions<4, f64>,
	) -> Result<f64> {
		let (nodes, leaves_end) = tree.nodes_to_update();
		let (nodes, children) = tree.to_lists(&nodes);
		let frequencies = transitions.frequencies();
		let tms = transitions.matrices(&nodes[..nodes.len() - 1]);

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
				tms.as_ptr() as *mut f64,
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

		for ((likelihood, scale), weight) in self
			.likelihoods
			.iter_mut()
			.zip(&self.scale_sums)
			.zip(&self.pattern_weights)
		{
			*likelihood -= f64::from(*scale);
			*likelihood *= f64::from(*weight);
		}

		Ok(self.likelihoods.iter().sum())
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

	fn num_patterns(&self) -> usize {
		self.num_patterns
	}
}

impl Cpu4 {
	pub fn new(
		pattern_weights: Vec<u32>,
		leaves: Vec<u8>,
		scale_ln: u32,
	) -> Self {
		let num_patterns = pattern_weights.len();
		let num_leaves = leaves.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;

		let projections =
			Buffer::<_, 32>::new(num_nodes * num_patterns * 2);
		let scales = vec![false; num_nodes * num_patterns * 2];
		let scale_sums = vec![0; num_patterns];

		let scale_ln_f64: f64 = scale_ln.into();
		let scale_mult = scale_ln_f64.exp();
		let scale_threshold = (-scale_ln_f64).exp();

		Self {
			num_patterns,
			pattern_weights,

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
