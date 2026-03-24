#![expect(clippy::undocumented_unsafe_blocks)]

use anyhow::Result;
use fork_union::{SyncConstPtr, SyncMutPtr, ThreadPool, count_logical_cores};
use parking_lot::MutexGuard;

use crate::{Transitions, likelihood::Calculator, parameters::Tree};
use buffer::Buffer;
use sk::EditBuf;

#[allow(dead_code)]
pub struct Cpu4Propagations {
	num_patterns: usize,

	pattern_weights: Vec<u32>,

	selectors: EditBuf,

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
		mut tree: MutexGuard<Tree>,
		transitions: &Transitions<4, f64>,
	) -> Result<f64> {
		let (nodes, children, leaves_end) = tree.propagation_lists();
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
				leaves_end,
				&tms,
				frequencies,
			)
		}

		for ((likelihood, scale), weight) in self
			.likelihoods
			.iter_mut()
			.zip(&self.scale_sums)
			.zip(&self.pattern_weights)
		{
			*likelihood -= f64::from(*scale * self.scale_ln);
			*likelihood *= f64::from(*weight);
		}

		Ok(self.likelihoods.iter().sum())
	}

	fn accept(&mut self) -> Result<()> {
		self.scale_sums_backup.copy_from_slice(&self.scale_sums);
		self.selectors.accept();

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.scale_sums.copy_from_slice(&self.scale_sums_backup);
		self.selectors.reject();
		Ok(())
	}

	fn num_patterns(&self) -> usize {
		self.num_patterns
	}
}

impl Cpu4Propagations {
	fn set_selectors(&mut self, nodes: &[usize]) {
		for &node in nodes {
			self.selectors.set_edited(node);
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

			selectors: EditBuf::new(num_nodes),

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
	children: &[[usize; 2]],
	leaves_end: usize,
	transitions: &[[[f64; 4]; 4]],
	frequencies: [f64; 4],
) {unsafe {
	let num_patterns = state.num_patterns;
	let selectors = &state.selectors;
	let samples = state.samples.as_ptr();
	let propagations = state.propagations.as_mut_ptr();

	let propagations_sync = SyncMutPtr::new(propagations);
	let samples_sync = SyncConstPtr::new(samples);
	let scale_sums_sync = SyncMutPtr::new(state.scale_sums.as_mut_ptr());
	let scales_sync = SyncMutPtr::new(state.scales.as_mut_ptr());

	macro_rules! offset {
		($index:expr) => {{
			let ptr = selectors.offset($index);
			($index * 2 + ptr) * num_patterns
		}};
	}
	macro_rules! offset_old {
		($index:expr) => {{
			let ptr = selectors.offset_other($index);
			($index * 2 + ptr) * num_patterns
		}};
	}

	let _ = state.pool.for_slices(num_patterns, |prong, count| {
	let start = prong.task_index;

	let propagations = propagations_sync.as_ptr();
	let samples = samples_sync.as_ptr();
	let scale_sums = scale_sums_sync.as_ptr();
	let scales = scales_sync.as_ptr();

	for (i, &leaf) in nodes.iter().enumerate().take(leaves_end) {
		let samples_leaf = samples.add(leaf * num_patterns);
		let propagations_leaf = propagations.add(offset!(leaf) + start);

		calc_leaf(
			count,
			&transitions[i],
			samples_leaf,
			propagations_leaf,
		);
	}

	for i in leaves_end..nodes.len() - 1 {
		let node = nodes[i];

		let [left, right] = children[i - leaves_end];

		let propagations_left = propagations.add(offset!(left) + start);
		let propagations_right = propagations.add(offset!(right) + start);
		let propagations_node = propagations.add(offset!(node) + start);

		let scales_new = scales.add(offset!(node) + start);
		let scales_old = scales.add(offset_old!(node) + start);

		calc_propagation(
			count,
			&transitions[i],
			propagations_left as *mut f64,
			propagations_right as *mut f64,
			propagations_node as *mut f64,
			scales_new,
			scales_old,
			scale_sums.add(start),
			state.scale_threshold,
			state.scale_mult,
		);
	}

	});


	let likelihoods = state.likelihoods.as_mut_ptr();

	let root = nodes.last().unwrap();
	let &[root_left, root_right] = children.last().unwrap();
	let mut propagations_left =
		propagations.add(offset!(root_left));
	let mut propagations_right =
		propagations.add(offset!(root_right));

	let mut scales_old = scales_sync.as_ptr().add(offset_old!(*root));
	let mut scales_new = scales_sync.as_ptr().add(offset!(*root));
	let mut scale_sums = scale_sums_sync.as_ptr();

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

		if *scales_old {
			*scale_sums -= 1;
		}
		*scales_new = false;

		scales_new = scales_new.add(1);
		scales_old = scales_old.add(1);
		scale_sums = scale_sums.add(1);
	}
}}

#[rustfmt::skip]
unsafe fn calc_leaf(
	num_patterns: usize,
	transition: &[[f64; 4]; 4],
	mut samples: *const u8,
	mut propagations: *mut [f64; 4],
) {unsafe {
	let t = *transition;

	for _ in 0..num_patterns {
		let sample = samples.read();

		if sample == 0b0001 {
			propagations.write([t[0][0], t[1][0], t[2][0], t[3][0]]);
		} else if sample == 0b0010 {
			propagations.write([t[0][1], t[1][1], t[2][1], t[3][1]]);
		} else if sample == 0b0100 {
			propagations.write([t[0][2], t[1][2], t[2][2], t[3][2]]);
		} else if sample == 0b1000 {
			propagations.write([t[0][3], t[1][3], t[2][3], t[3][3]]);
		} else {
			propagations.write([1.0, 1.0, 1.0, 1.0]);
		}

		propagations = propagations.add(1);
		samples = samples.add(1);
	}
}}

#[rustfmt::skip]
#[expect(clippy::too_many_arguments)]
unsafe fn calc_propagation(
	num_patterns: usize,
	transition: &[[f64; 4]; 4],
	mut propagations_left: *const f64,
	mut propagations_right: *const f64,
	mut propagations_node: *mut f64,
	mut scales: *mut bool,
	mut scales_old: *mut bool,
	mut scale_sums: *mut u32,
	threshold: f64,
	mult: f64,
) {unsafe {
	let t = *transition;

	for _ in 0..num_patterns {
		let mut p0 = propagations_left.add(0).read()
			* propagations_right.add(0).read();
		let mut p1 = propagations_left.add(1).read()
			* propagations_right.add(1).read();
		let mut p2 = propagations_left.add(2).read()
			* propagations_right.add(2).read();
		let mut p3 = propagations_left.add(3).read()
			* propagations_right.add(3).read();

		let old_scale = scales_old.read();

		if p0 < threshold && p1 < threshold && p2 < threshold && p3 < threshold {
			p0 *= mult;
			p1 *= mult;
			p2 *= mult;
			p3 *= mult;

			if !old_scale {
				*scale_sums += 1;
			}
			*scales = true;
		} else {
			if old_scale {
				*scale_sums -= 1;
			}
			*scales = false;
		}

		propagations_node
			.add(0)
			.write(p0 * t[0][0] + p1 * t[0][1] + p2 * t[0][2] + p3 * t[0][3]);
		propagations_node
			.add(1)
			.write(p0 * t[1][0] + p1 * t[1][1] + p2 * t[1][2] + p3 * t[1][3]);
		propagations_node
			.add(2)
			.write(p0 * t[2][0] + p1 * t[2][1] + p2 * t[2][2] + p3 * t[2][3]);
		propagations_node
			.add(3)
			.write(p0 * t[3][0] + p1 * t[3][1] + p2 * t[3][2] + p3 * t[3][3]);

		propagations_node = propagations_node.add(4);
		propagations_left = propagations_left.add(4);
		propagations_right = propagations_right.add(4);
		scales_old = scales_old.add(1);
		scales = scales.add(1);
		scale_sums = scale_sums.add(1);
	}
}}
