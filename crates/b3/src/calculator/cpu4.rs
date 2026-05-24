#![expect(clippy::undocumented_unsafe_blocks)]
#![expect(clippy::too_many_arguments)]

use anyhow::Result;
use bytemuck::{cast_slice, checked::cast_slice_mut};
use fork_union::{SyncConstPtr, SyncMutPtr, ThreadPool, count_logical_cores};

use crate::{Transitions, calculator::Calculator, parameters::Tree};
use buffer::Buffer;
use linalg::Vector;
use sk::EditBuf;

pub struct Cpu4Calculator {
	num_leaves: usize,
	num_patterns: usize,

	pattern_weights: Vec<u32>,

	selectors: EditBuf,

	samples: Vec<u8>,
	/// 32-bit alignment for aligned loads into YMM registers.
	partials: Buffer<[f64; 4], 32>,

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

impl Calculator<f64> for Cpu4Calculator {
	fn likelihood(
		&mut self,
		tree: &Tree,
		transitions: &Transitions,
	) -> Result<f64> {
		let frequencies =
			*transitions.frequencies().as_array::<4>().unwrap();

		let (internals, children) = tree.partials_lists();
		let mut matrices = vec![[[0.0; 4]; 4]; children.len() * 2];
		transitions.write_matrices(
			cast_slice(&children),
			cast_slice_mut(&mut matrices),
		);
		let tms: Vec<[[f64; 5]; 4]> = matrices
			.into_iter()
			.map(|matrix| {
				matrix.map(|row| {
					let [a, c, g, t] = row;
					[a, c, g, t, 1.0]
				})
			})
			.collect();

		self.set_selectors(&internals);

		unsafe {
			propose(self, &internals, &children, &tms, frequencies)
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

	fn likelihoods(&self) -> &[f64] {
		&self.likelihoods
	}

	fn num_patterns(&self) -> usize {
		self.num_patterns
	}
}

impl Cpu4Calculator {
	fn set_selectors(&mut self, nodes: &[usize]) {
		for &node in nodes {
			self.selectors.set_edited(node - self.num_leaves);
		}
	}

	pub fn new(
		pattern_weights: Vec<u32>,
		mut samples: Vec<u8>,
		scale_ln: u32,
		num_threads: usize,
	) -> Self {
		let num_patterns = pattern_weights.len();
		let num_leaves = samples.len() / num_patterns;
		let num_internals = num_leaves - 1;

		for sample in &mut samples {
			*sample = match *sample {
				0b0001 => 0,
				0b0010 => 1,
				0b0100 => 2,
				0b1000 => 3,
				_ => 4,
			}
		}

		let num_threads = if num_threads == 0 {
			let num_cores = count_logical_cores();
			num_cores.min(num_patterns.div_ceil(1000))
		} else {
			num_threads
		};

		let partials =
			Buffer::<_, 32>::new(num_internals * num_patterns * 2);
		let scales = vec![false; num_internals * num_patterns * 2];
		let scale_sums = vec![0; num_patterns];

		let scale_ln_f64: f64 = scale_ln.into();
		let scale_mult = scale_ln_f64.exp();
		let scale_threshold = (-scale_ln_f64).exp();

		Self {
			num_leaves,
			num_patterns,
			pattern_weights,

			selectors: EditBuf::new(num_internals),

			samples,
			partials,

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
	state: &mut Cpu4Calculator,
	internals: &[usize],
	children: &[[usize; 2]],
	transitions: &[[[f64; 5]; 4]],
	frequencies: [f64; 4],
) {unsafe {
	let num_patterns = state.num_patterns;
	let selectors = &state.selectors;
	let samples = state.samples.as_ptr();
	let partials = state.partials.as_mut_ptr();

	let partials_sync = SyncMutPtr::new(partials);
	let samples_sync = SyncConstPtr::new(samples);
	let scale_sums_sync = SyncMutPtr::new(state.scale_sums.as_mut_ptr());
	let scales_sync = SyncMutPtr::new(state.scales.as_mut_ptr());

	macro_rules! offset {
		($index:expr) => {{
			let ptr = selectors.offset_unchecked($index);
			($index * 2 + ptr) * num_patterns
		}};
	}
	macro_rules! offset_old {
		($index:expr) => {{
			let ptr = selectors.offset_other_unchecked($index);
			($index * 2 + ptr) * num_patterns
		}};
	}

	let _ = state.pool.for_slices(num_patterns, |prong, count| {
	let start = prong.task_index;

	let partials = partials_sync.as_ptr();
	let samples = samples_sync.as_ptr();
	let scale_sums = scale_sums_sync.as_ptr();
	let scales = scales_sync.as_ptr();

	for (i, &internal) in internals.iter().enumerate() {
		let [left, right] = children[i];  // left < right
		let partials_internal = partials.add(offset!(internal - state.num_leaves) + start);

		let transition_left = &transitions[i * 2];
		let transition_right = &transitions[i * 2 + 1];

		let scales_new = scales.add(offset!(internal - state.num_leaves) + start);
		let scales_old = scales.add(offset_old!(internal - state.num_leaves) + start);

		if right < state.num_leaves { // both are children
			let samples_left = samples.add(left * num_patterns + start);
			let samples_right = samples.add(right * num_patterns + start);

			calc_sample_sample(
				count,
				partials_internal,
				samples_left,
				samples_right,
				transition_left,
				transition_right,

				scales_new,
				scales_old,
				scale_sums.add(start),
			);
		} else if left < state.num_leaves && right >= state.num_leaves {
			let samples_left = samples.add(left * num_patterns + start);
			let partials_right = partials.add(offset!(right - state.num_leaves) + start);

			calc_sample_partial(
				count,
				partials_internal,
				samples_left,
				partials_right,
				transition_left,
				transition_right,

				scales_new,
				scales_old,
				scale_sums.add(start),
				state.scale_threshold,
				state.scale_mult,
			);
		} else {
			let partials_left = partials.add(offset!(left - state.num_leaves) + start);
			let partials_right = partials.add(offset!(right - state.num_leaves) + start);

			calc_partial_partial(
				count,
				partials_internal,
				partials_left,
				partials_right,
				transition_left,
				transition_right,

				scales_new,
				scales_old,
				scale_sums.add(start),
				state.scale_threshold,
				state.scale_mult,
			);
		}
	}

	});

	let root = *internals.last().unwrap();
	let partials_root = partials.add(offset!(root - state.num_leaves));

	for i in 0..num_patterns {
		let partial = partials_root.add(i).read();
		let prod: [f64; 4] = partial.hadamard(frequencies);
		state.likelihoods[i] = prod.sum().ln();
	}
}}

#[rustfmt::skip]
unsafe fn calc_sample_sample(
	num_patterns: usize,
	partials: *mut [f64; 4],
	samples_left: *const u8,
	samples_right: *const u8,
	transition_left: &[[f64; 5]; 4],
	transition_right: &[[f64; 5]; 4],

	scales: *mut bool,
	scales_old: *mut bool,
	scale_sums: *mut u32,
) {unsafe {
	let tl = *transition_left;
	let tr = *transition_right;

	for i in 0..num_patterns {
		let sample_left = samples_left.add(i).read() as usize;
		let sample_right = samples_right.add(i).read() as usize;

		let scales_old = scales_old.add(i).read();
		if scales_old {
			*scale_sums.add(i) -= 1;
		}
		scales.add(i).write(false);

		let out = partials.add(i);
		for j in 0..4 {
			(*out)[j] = tl[j][sample_left] * tr[j][sample_right];
		}
	}
}}

#[rustfmt::skip]
unsafe fn calc_sample_partial(
	num_patterns: usize,
	partials: *mut [f64; 4],
	samples_left: *const u8,
	partials_right: *const [f64; 4],
	transition_left: &[[f64; 5]; 4],
	transition_right: &[[f64; 5]; 4],

	scales: *mut bool,
	scales_old: *mut bool,
	scale_sums: *mut u32,
	threshold: f64,
	mult: f64,
) { unsafe {
	let tl = *transition_left;
	let tr = *transition_right;

	for i in 0..num_patterns {
		let sample_left = samples_left.add(i).read() as usize;
		let partial_right = partials_right.add(i).read();

		let mut res_r = [0.0; 4];
		for row in 0..4 {
			for col in 0..4 {
				res_r[row] += tr[row][col] * partial_right[col];
			}
		}

		let mut prod = [0.0; 4];
		for j in 0..4 {
			prod[j] = tl[j][sample_left] * res_r[j];
		}

		scale(i, &mut prod, scales, scales_old, scale_sums, threshold, mult);

		partials.add(i).write(prod);
	}
}}

#[rustfmt::skip]
unsafe fn calc_partial_partial(
	num_patterns: usize,
	partials: *mut [f64; 4],
	partials_left: *const [f64; 4],
	partials_right: *const [f64; 4],
	transition_left: &[[f64; 5]; 4],
	transition_right: &[[f64; 5]; 4],

	scales: *mut bool,
	scales_old: *mut bool,
	scale_sums: *mut u32,
	threshold: f64,
	mult: f64,
) {unsafe {
	let tl = *transition_left;
	let tr = *transition_right;

	for i in 0..num_patterns {
		let mut res_l = [0.0; 4];
		let mut res_r = [0.0; 4];

		let partial_left = partials_left.add(i).read();
		let partial_right = partials_right.add(i).read();

		for row in 0..4 {
			for col in 0..4 {
				res_l[row] += tl[row][col] * partial_left[col];
				res_r[row] += tr[row][col] * partial_right[col];
			}
		}

		let mut prod = [0.0; 4];
		for j in 0..4 {
			prod[j] = res_l[j] * res_r[j];
		}

		scale(i, &mut prod, scales, scales_old, scale_sums, threshold, mult);

		partials.add(i).write(prod);
	}
}}

unsafe fn scale(
	i: usize,
	prod: &mut [f64; 4],
	scales: *mut bool,
	scales_old: *mut bool,
	scale_sums: *mut u32,
	threshold: f64,
	mult: f64,
) {
	unsafe {
		let old_scale = scales_old.add(i).read();
		if prod[0] < threshold
			&& prod[1] < threshold
			&& prod[2] < threshold
			&& prod[3] < threshold
		{
			for el in prod {
				*el *= mult;
			}
			if !old_scale {
				*scale_sums.add(i) += 1;
			}
			*scales.add(i) = true;
		} else {
			if old_scale {
				*scale_sums.add(i) -= 1;
			}
			*scales.add(i) = false;
		}
	}
}
