#![expect(clippy::undocumented_unsafe_blocks)]

unsafe fn calc_leaf(
	num_patterns: usize,
	transition: *const f64,
	mut samples: *const u8,
	mut propagations: *mut f64,
) {
	unsafe {
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
	}
}

unsafe fn calc_propagation(
	num_patterns: usize,
	transition: *const f64,
	mut propagations_left: *const f64,
	mut propagations_right: *const f64,
	mut propagations_node: *mut f64,
) {
	unsafe {
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
				.write(p0 * m_00
					+ p1 * m_01 + p2 * m_02 + p3 * m_03);
			propagations_node
				.add(1)
				.write(p0 * m_10
					+ p1 * m_11 + p2 * m_12 + p3 * m_13);
			propagations_node
				.add(2)
				.write(p0 * m_20
					+ p1 * m_21 + p2 * m_22 + p3 * m_23);
			propagations_node
				.add(3)
				.write(p0 * m_30
					+ p1 * m_31 + p2 * m_32 + p3 * m_33);

			propagations_node = propagations_node.add(4);
			propagations_left = propagations_left.add(4);
			propagations_right = propagations_right.add(4);
		}
	}
}

#[rustfmt::skip]
#[expect(clippy::too_many_arguments)]
pub unsafe fn propose(
	nodes: &[usize],
	children: &[(usize, usize)],
	transition: *const f64,
	leaves_end: usize,
	frequencies: [f64; 4],
	//
	num_patterns: usize,
	samples: *const u8,
	propagations: *mut f64,
	selectors: *const u8,
	likelihoods: *mut f64,
) {
	macro_rules! offset {
		($index:expr) => {{
			let selector = selectors.add($index).read();
			let fix = (selector >> 1) ^ (selector & 0b01);
			($index * 2 + fix as usize) * num_patterns * 4
		}};
	}

	unsafe {

	for (i, &leaf) in nodes.iter().enumerate().take(leaves_end) {
		let transition = transition.add(i * 16);
		let samples_leaf = samples.add(leaf * num_patterns);
		let propagations_leaf = propagations.add(offset!(leaf));

		calc_leaf(
			num_patterns,
			transition,
			samples_leaf,
			propagations_leaf,
		);
	}

	for i in leaves_end..nodes.len() - 1 {
		let node = nodes[i];

		let transition = transition.add(i * 16);
		let (left, right) = children[i - leaves_end];

		let propagations_left = propagations.add(offset!(left));
		let propagations_right = propagations.add(offset!(right));
		let propagations_node = propagations.add(offset!(node));

		calc_propagation(
			num_patterns,
			transition,
			propagations_left,
			propagations_right,
			propagations_node,
		);
	}

	let &(root_left, root_right) = children.last().unwrap();
	let mut propagations_left =
		propagations.add(offset!(root_left));
	let mut propagations_right =
		propagations.add(offset!(root_right));

	for i in 0..num_patterns {
		let p0 = propagations_left.add(0).read()
			* propagations_right.add(0).read();
		let p1 = propagations_left.add(1).read()
			* propagations_right.add(1).read();
		let p2 = propagations_left.add(2).read()
			* propagations_right.add(2).read();
		let p3 = propagations_left.add(3).read()
			* propagations_right.add(3).read();

		let result = (
			p0 * frequencies[0] +
			p1 * frequencies[1] +
			p2 * frequencies[2] +
			p3 * frequencies[3]
		).ln();
		likelihoods.add(i).write(result);

		propagations_left = propagations_left.add(4);
		propagations_right = propagations_right.add(4);
	}

	}
}
