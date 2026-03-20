#![expect(clippy::undocumented_unsafe_blocks)]

use anyhow::Result;
use fork_union::{SyncConstPtr, SyncMutPtr, ThreadPool, count_logical_cores};
use parking_lot::MutexGuard;

use crate::{Transitions, likelihood::Calculator, parameters::Tree};
use buffer::Buffer;

#[allow(dead_code)]
pub struct Cpu4Propagations {
	num_patterns: usize,

	pattern_weights: Vec<u32>,

	selectors: Vec<u8>,

	/// Have the length of `num_patterns`
	samples: Vec<u8>,
	/// 32-bit alignment for aligned loads into YMM registers.
	propagations: Buffer<[f64; 4], 32>,

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

	pool: ThreadPool,
}

impl Calculator<4, f64> for Cpu4Propagations {
	fn likelihood(
		&mut self,
		tree: MutexGuard<Tree>,
		transitions: &Transitions<4, f64>,
	) -> Result<f64> {
		let (nodes, leaves_end) = tree.nodes_to_update();
		let (nodes, children) = tree.to_lists(&nodes);
		let frequencies = transitions.frequencies();
		let tms = transitions.matrices(&nodes[..nodes.len() - 1]);

		drop(tree);

		self.set_selectors(&nodes);

		// SAFETY: single threaded, we own the buffers
		unsafe {
			propose(
				self,
				&nodes,
				&children,
				tms.as_ptr() as *const f64,
				leaves_end,
				frequencies,
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
		self.scale_sums_backup.copy_from_slice(&self.scale_sums);
		// TODO: op transform
		for selector in &mut self.selectors {
			*selector = match *selector {
				0b00 => 0b00,
				0b01 => 0b01,
				0b10 => 0b01,
				0b11 => 0b00,
				_ => unreachable!(),
			}
		}

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.scale_sums.copy_from_slice(&self.scale_sums_backup);
		// TODO: op transform
		for selector in &mut self.selectors {
			*selector = match *selector {
				0b00 => 0b00,
				0b01 => 0b01,
				0b10 => 0b00,
				0b11 => 0b01,
				_ => unreachable!(),
			}
		}

		Ok(())
	}

	fn num_patterns(&self) -> usize {
		self.num_patterns
	}
}

impl Cpu4Propagations {
	fn set_selectors(&mut self, nodes: &[usize]) {
		for node in nodes {
			self.selectors[*node] |= 0b10;
		}
	}

	pub fn new(
		pattern_weights: Vec<u32>,
		samples: Vec<u8>,
		num_threads: usize,
		scale_ln: u32,
	) -> Self {
		let num_patterns = pattern_weights.len();
		let num_leaves = samples.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;

		let num_threads = if num_threads == 0 {
			let num_cores = count_logical_cores();
			num_cores.min(num_patterns.div_ceil(1000))
		} else {
			num_threads
		};

		let propagations =
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

			samples,
			propagations,

			likelihoods: vec![f64::NAN; num_patterns],

			scales,
			scale_sums_backup: scale_sums.clone(),
			scale_sums,

			scale_ln,
			scale_threshold,
			scale_mult,

			pool: ThreadPool::try_spawn(num_threads).unwrap(),
		}
	}
}

#[inline(never)]
#[rustfmt::skip]
unsafe fn propose(
	state: &mut Cpu4Propagations,
	nodes: &[usize],
	children: &[(usize, usize)],
	transitions: *const f64,
	leaves_end: usize,
	frequencies: [f64; 4],
) {unsafe {
	let num_patterns = state.num_patterns;
	let selectors = &state.selectors;
	let samples = state.samples.as_ptr();
	let propagations = state.propagations.as_mut_ptr();

	let transitions_sync = SyncConstPtr::new(transitions);
	let propagations_sync = SyncMutPtr::new(propagations);
	let samples_sync = SyncConstPtr::new(samples);

	macro_rules! offset {
		($index:expr) => {{
			let selector = selectors[$index];
			let fix = (selector >> 1) ^ (selector & 0b01);
			($index * 2 + fix as usize) * num_patterns
		}};
	}

	let _ = state.pool.for_slices(num_patterns, |prong, count| {
	let start = prong.task_index;

	let propagations = propagations_sync.as_ptr();
	let samples = samples_sync.as_ptr();
	let transitions = transitions_sync.as_ptr();

	for (i, &leaf) in nodes.iter().enumerate().take(leaves_end) {
		let transition = transitions.add(i * 16);
		let samples_leaf = samples.add(leaf * num_patterns);
		let propagations_leaf = propagations.add(offset!(leaf) + start);

		calc_leaf(
			count,
			transition,
			samples_leaf,
			propagations_leaf as *mut f64,
		);
	}

	for i in leaves_end..nodes.len() - 1 {
		let node = nodes[i];

		let transition = transitions.add(i * 16);
		let (left, right) = children[i - leaves_end];

		let propagations_left = propagations.add(offset!(left) + start);
		let propagations_right = propagations.add(offset!(right) + start);
		let propagations_node = propagations.add(offset!(node) + start);

		calc_propagation(
			count,
			transition,
			propagations_left as *mut f64,
			propagations_right as *mut f64,
			propagations_node as *mut f64,
		);
	}

	});


	let likelihoods = state.likelihoods.as_mut_ptr();

	let &(root_left, root_right) = children.last().unwrap();
	let mut propagations_left =
		propagations.add(offset!(root_left));
	let mut propagations_right =
		propagations.add(offset!(root_right));

	for i in 0..num_patterns {
		let pl = propagations_left.read();
		let pr = propagations_right.read();

		let result = (
			pl[0] * pr[0] * frequencies[0] +
			pl[1] * pr[1] * frequencies[1] +
			pl[2] * pr[2] * frequencies[2] +
			pl[3] * pr[3] * frequencies[3]
		).ln();
		likelihoods.add(i).write(result);

		propagations_left = propagations_left.add(1);
		propagations_right = propagations_right.add(1);
	}
}}

#[rustfmt::skip]
unsafe fn calc_leaf(
	num_patterns: usize,
	transition: *const f64,
	mut samples: *const u8,
	mut propagations: *mut f64,
) {unsafe {
	let m_00 = transition.add(0).read();
	let m_01 = transition.add(1).read();
	let m_02 = transition.add(2).read();
	let m_03 = transition.add(3).read();
	let m_10 = transition.add(4).read();
	let m_11 = transition.add(5).read();
	let m_12 = transition.add(6).read();
	let m_13 = transition.add(7).read();
	let m_20 = transition.add(8).read();
	let m_21 = transition.add(9).read();
	let m_22 = transition.add(10).read();
	let m_23 = transition.add(11).read();
	let m_30 = transition.add(12).read();
	let m_31 = transition.add(13).read();
	let m_32 = transition.add(14).read();
	let m_33 = transition.add(15).read();

	for _ in 0..num_patterns {
		let sample = samples.read();

		if sample == 0b0001 {
			propagations.add(0).write(m_00);
			propagations.add(1).write(m_10);
			propagations.add(2).write(m_20);
			propagations.add(3).write(m_30);
		} else if sample == 0b0010 {
			propagations.add(0).write(m_01);
			propagations.add(1).write(m_11);
			propagations.add(2).write(m_21);
			propagations.add(3).write(m_31);
		} else if sample == 0b0100 {
			propagations.add(0).write(m_02);
			propagations.add(1).write(m_12);
			propagations.add(2).write(m_22);
			propagations.add(3).write(m_32);
		} else if sample == 0b1000 {
			propagations.add(0).write(m_03);
			propagations.add(1).write(m_13);
			propagations.add(2).write(m_23);
			propagations.add(3).write(m_33);
		} else {
			propagations.add(0).write(1.0);
			propagations.add(1).write(1.0);
			propagations.add(2).write(1.0);
			propagations.add(3).write(1.0);
		}

		propagations = propagations.add(4);
		samples = samples.add(1);
	}
}}

#[rustfmt::skip]
unsafe fn calc_propagation(
	num_patterns: usize,
	transition: *const f64,
	mut propagations_left: *const f64,
	mut propagations_right: *const f64,
	mut propagations_node: *mut f64,
) {unsafe {
	let m_00 = transition.add(0).read();
	let m_01 = transition.add(1).read();
	let m_02 = transition.add(2).read();
	let m_03 = transition.add(3).read();
	let m_10 = transition.add(4).read();
	let m_11 = transition.add(5).read();
	let m_12 = transition.add(6).read();
	let m_13 = transition.add(7).read();
	let m_20 = transition.add(8).read();
	let m_21 = transition.add(9).read();
	let m_22 = transition.add(10).read();
	let m_23 = transition.add(11).read();
	let m_30 = transition.add(12).read();
	let m_31 = transition.add(13).read();
	let m_32 = transition.add(14).read();
	let m_33 = transition.add(15).read();

	for _ in 0..num_patterns {
		let p0 = propagations_left.add(0).read()
			* propagations_right.add(0).read();
		let p1 = propagations_left.add(1).read()
			* propagations_right.add(1).read();
		let p2 = propagations_left.add(2).read()
			* propagations_right.add(2).read();
		let p3 = propagations_left.add(3).read()
			* propagations_right.add(3).read();

		propagations_node
			.add(0)
			.write(p0 * m_00 + p1 * m_01 + p2 * m_02 + p3 * m_03);
		propagations_node
			.add(1)
			.write(p0 * m_10 + p1 * m_11 + p2 * m_12 + p3 * m_13);
		propagations_node
			.add(2)
			.write(p0 * m_20 + p1 * m_21 + p2 * m_22 + p3 * m_23);
		propagations_node
			.add(3)
			.write(p0 * m_30 + p1 * m_31 + p2 * m_32 + p3 * m_33);

		propagations_node = propagations_node.add(4);
		propagations_left = propagations_left.add(4);
		propagations_right = propagations_right.add(4);
	}
}}
