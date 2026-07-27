#![allow(unused)]

use anyhow::Result;
use parking_lot::MutexGuard;

use crate::{Transitions, parameters::Tree};
use buffer::Buffer;
use sk::EditBuf;

pub struct StateCalculator {
	num_leaves: u32,

	edits: EditBuf,

	values: Vec<u8>,
	partials: Buffer<f64, 32>,

	likelihood: f64,

	scales: Vec<bool>,
	scale_sum: u32,
}

impl StateCalculator {
	pub fn likelihood(
		&mut self,
		mut tree: MutexGuard<Tree>,
		transitions: &Transitions,
	) -> Result<f64> {
		let (internals, children) = tree.partials_lists();
		drop(tree);

		self.set_selectors(&internals);

		todo!()
	}

	pub fn accept(&mut self) -> Result<()> {
		self.edits.accept();

		Ok(())
	}

	pub fn reject(&mut self) -> Result<()> {
		self.edits.reject();
		Ok(())
	}

	fn set_selectors(&mut self, nodes: &[u32]) {
		for &node in nodes {
			let idx = node - self.num_leaves;
			self.edits.set_edited(idx as usize);
		}
	}

	pub fn new(size: usize, values: Vec<u8>) -> Self {
		let num_leaves = values.len();
		let num_internals = num_leaves - 1;

		let partials =
			Buffer::<_, 32>::new(size * size * num_internals * 2);
		let scales = vec![false; num_internals * 2];

		Self {
			num_leaves: num_leaves as u32,

			edits: EditBuf::new(num_internals),

			values,
			partials,

			scale_sum: 0,

			likelihood: f64::NAN,

			scales,
		}
	}
}
