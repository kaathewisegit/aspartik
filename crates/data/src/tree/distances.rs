use anyhow::{Result, ensure};
use hashbrown::HashTable;
use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::SmallVec;

use std::ptr;

use super::{BinaryTree, Node};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CladeHash {
	first: u64,
	second: u64,
	size: u32,
}

impl CladeHash {
	fn leaf(index: u32) -> Self {
		let index = u64::from(index) + 1;
		Self {
			first: mix(index ^ 0x243f_6a88_85a3_08d3),
			second: mix(index ^ 0x1319_8a2e_0370_7344),
			size: 1,
		}
	}

	fn combine(self, other: Self) -> Self {
		Self {
			first: self.first.wrapping_add(other.first),
			second: self.second.wrapping_add(other.second),
			size: self.size + other.size,
		}
	}
}

fn mix(mut value: u64) -> u64 {
	value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
	value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	value ^ (value >> 31)
}

fn clade_hashes(tree: &BinaryTree) -> Vec<CladeHash> {
	let mut hashes = vec![CladeHash::default(); tree.num_nodes() as usize];
	for node in tree.postorder() {
		let hash = clade_hash(tree, &hashes, node);
		hashes[node.usize()] = hash;
	}
	hashes
}

pub(super) fn robinson_foulds(first: &BinaryTree, second: &BinaryTree) -> u32 {
	assert_eq!(first.num_leaves(), second.num_leaves());
	let first_hashes = clade_hashes(first);
	let second_hashes = clade_hashes(second);
	let mut clades =
		HashTable::with_capacity(first.num_leaves() as usize - 2);
	for node in first.internals().filter(|&node| node != first.root()) {
		let hash = first_hashes[node.usize()];
		clades.insert_unique(hash.first, hash, |hash| hash.first);
	}
	let shared = second
		.internals()
		.filter(|&node| node != second.root())
		.filter(|&node| {
			let hash = second_hashes[node.usize()];
			clades.find(hash.first, |clade| *clade == hash)
				.is_some()
		})
		.count() as u32;
	2 * (first.num_leaves() - 2 - shared)
}

fn clade_hash(
	tree: &BinaryTree,
	hashes: &[CladeHash],
	node: Node,
) -> CladeHash {
	if let Some(leaf) = tree.as_leaf(node) {
		CladeHash::leaf(leaf.u32())
	} else {
		let internal = tree.as_internal(node).unwrap();
		let [left, right] = tree.children_of(internal);
		hashes[left.usize()].combine(hashes[right.usize()])
	}
}

/// Computes all rooted [Robinson-Foulds distances][rf] by indexing shared
/// clades
///
/// Expected time is `O(kn + sum(f_c^2))` for `k` trees with `n` leaves, where
/// `f_c` is the number of trees containing clade `c`.
///
/// [rf]: https://doi.org/10.1016/0025-5564(81)90043-2
pub fn robinson_foulds_matrix(trees: &[&BinaryTree]) -> Result<Vec<Vec<u32>>> {
	let Some(first_tree) = trees.first() else {
		return Ok(Vec::new());
	};

	let num_leaves = first_tree.num_leaves();
	for tree in trees[1..].iter() {
		ensure!(
			tree.num_leaves() == num_leaves,
			"Expected every tree to have {num_leaves} leaves, got {}",
			tree.num_leaves()
		);
		ensure!(
			first_tree.identical_children(tree),
			"Expected every tree to use the same leaf IDs"
		);
	}

	let num_clades = usize::try_from(num_leaves - 2)?;
	let num_nodes = first_tree.num_nodes() as usize;
	let capacity = trees.len() * num_clades;

	let mut clades = Vec::with_capacity(capacity);
	let mut hashes = vec![CladeHash::default(); num_nodes];

	for (tree_index, tree) in trees.iter().enumerate() {
		if hashes.len() < num_nodes {
			hashes.resize(num_nodes, CladeHash::default());
		}
		for node in tree.postorder() {
			let hash = clade_hash(tree, &hashes, node);
			hashes[node.usize()] = hash;
			if node != tree.root().into() && tree.is_internal(node)
			{
				clades.push((hash, tree_index));
			}
		}
	}

	clades.sort_unstable();

	let max_distance = (num_leaves - 2) * 2;
	let mut distances = vec![vec![max_distance; trees.len()]; trees.len()];
	for (index, row) in distances.iter_mut().enumerate() {
		row[index] = 0;
	}

	let mut slice_start = 0;
	while slice_start < clades.len() {
		let current_hash = clades[slice_start].0;
		let mut slice_end = slice_start + 1;
		while slice_end < clades.len()
			&& clades[slice_end].0 == current_hash
		{
			slice_end += 1;
		}

		// This is faster than collecting `clades` and then doing a
		// `dedup` on `distinct_trees`.
		let group = &clades[slice_start..slice_end];
		let mut distinct_trees: SmallVec<[usize; 8]> = SmallVec::new();
		for &(_, tree_idx) in group {
			if distinct_trees.last() != Some(&tree_idx) {
				distinct_trees.push(tree_idx);
			}
		}

		for (offset, &first) in distinct_trees.iter().enumerate() {
			for &second in &distinct_trees[offset + 1..] {
				distances[first][second] -= 2;
				distances[second][first] -= 2;
			}
		}

		slice_start = slice_end;
	}

	Ok(distances)
}

/// Computes the Kuhner-Felsenstein branch-score distance from hashed clades.
///
/// Expected time is O(n) for trees with n leaves. See
/// <https://doi.org/10.1093/oxfordjournals.molbev.a040126>.
pub fn branch_score(first: &BinaryTree, second: &BinaryTree) -> Result<f64> {
	ensure!(
		first.num_leaves() == second.num_leaves(),
		"Expected both trees to have {} leaves, got {}",
		first.num_leaves(),
		second.num_leaves()
	);
	let first_hashes = clade_hashes(first);
	let second_hashes = clade_hashes(second);
	let mut lengths = FxHashMap::with_capacity_and_hasher(
		first.num_edges() as usize,
		FxBuildHasher,
	);
	for child in first.edges() {
		lengths.insert(
			first_hashes[child.usize()],
			first.edge_length(child).unwrap(),
		);
	}

	let mut squared = 0.0;
	for child in second.edges() {
		let length = second.edge_length(child).unwrap();
		let first_length = lengths
			.remove(&second_hashes[child.usize()])
			.unwrap_or(0.0);
		squared += (first_length - length).powi(2);
	}
	squared += lengths.values().map(|length| length.powi(2)).sum::<f64>();
	Ok(squared.sqrt())
}

pub fn branch_score_matrix(trees: &[&BinaryTree]) -> Result<Vec<Vec<f64>>> {
	let Some(first_tree) = trees.first() else {
		return Ok(Vec::new());
	};

	let num_leaves = first_tree.num_leaves();
	for tree in &trees[1..] {
		ensure!(
			tree.num_leaves() == num_leaves,
			"Expected every tree to have {num_leaves} leaves, got {}",
			tree.num_leaves()
		);
		ensure!(
			first_tree.identical_children(tree),
			"Expected every tree to use the same leaf IDs"
		);
	}

	let capacity = trees.len() * first_tree.num_edges() as usize;
	let mut clades = Vec::with_capacity(capacity);
	let mut hashes =
		vec![CladeHash::default(); first_tree.num_nodes() as usize];

	for (tree_index, tree) in trees.iter().enumerate() {
		for node in tree.postorder() {
			let hash = clade_hash(tree, &hashes, node);
			hashes[node.usize()] = hash;
			if node != tree.root().into() {
				clades.push((hash, tree_index, node));
			}
		}
	}

	clades.sort_unstable();

	let mut norms = vec![0.0; trees.len()];
	let mut distances = vec![vec![0.0; trees.len()]; trees.len()];
	let mut start = 0;
	while start < clades.len() {
		let hash = clades[start].0;
		let mut end = start + 1;
		while end < clades.len() && clades[end].0 == hash {
			end += 1;
		}

		let group = &clades[start..end];
		for (offset, &(_, first, node)) in group.iter().enumerate() {
			let length = trees[first].edge_length(node).unwrap();
			norms[first] += length * length;
			for &(_, second, other_node) in &group[offset + 1..] {
				let other_length = trees[second]
					.edge_length(other_node)
					.unwrap();
				distances[first][second] +=
					length * other_length;
			}
		}
		start = end;
	}

	for first in 0..trees.len() {
		for second in first + 1..trees.len() {
			let squared = norms[first] + norms[second]
				- 2.0 * distances[first][second];
			let distance = if squared
				<= 8.0 * f64::EPSILON
					* (norms[first] + norms[second])
				|| !squared.is_finite()
			{
				if ptr::eq(trees[first], trees[second]) {
					0.0
				} else {
					branch_score(
						trees[first],
						trees[second],
					)?
				}
			} else {
				squared.sqrt()
			};
			distances[first][second] = distance;
			distances[second][first] = distance;
		}
	}

	Ok(distances)
}

impl BinaryTree {
	/// Computes the rooted Robinson-Foulds distance from hashed clades.
	///
	/// Expected time is O(n) for trees with n leaves. See
	/// <https://doi.org/10.1016/0025-5564(81)90043-2>.
	pub fn robinson_foulds(&self, other: &Self) -> u32 {
		robinson_foulds(self, other)
	}

	/// Computes triplet distance using cache-oblivious tree contractions.
	///
	/// Time is O(n log n) for trees with n leaves. See
	/// <https://doi.org/10.4230/LIPIcs.ESA.2017.21>.
	pub fn triplet_distance(&self, other: &Self) -> u128 {
		assert_eq!(self.num_leaves(), other.num_leaves());

		if self.num_leaves() < 3 {
			return 0;
		}

		TripletCounter::new(self, other).distance()
	}
}

/// Computes all pairwise triplet distances using tree contractions.
///
/// Time is O(k^2 n log n) for k trees with n leaves. See
/// <https://doi.org/10.4230/LIPIcs.ESA.2017.21>.
pub fn triplet_distance_matrix(
	trees: &[&BinaryTree],
) -> Result<Vec<Vec<u128>>> {
	let Some(first_tree) = trees.first() else {
		return Ok(Vec::new());
	};

	let num_leaves = first_tree.num_leaves();
	for tree in &trees[1..] {
		ensure!(
			tree.num_leaves() == num_leaves,
			"Expected every tree to have {num_leaves} leaves, got {}",
			tree.num_leaves()
		);
		ensure!(
			first_tree.identical_children(tree),
			"Expected every tree to use the same leaf IDs"
		);
	}

	let mut distances = vec![vec![0; trees.len()]; trees.len()];
	for first in 0..trees.len() {
		for second in 0..first {
			let distance =
				trees[first].triplet_distance(trees[second]);
			distances[first][second] = distance;
			distances[second][first] = distance;
		}
	}
	Ok(distances)
}

#[derive(Clone, Copy)]
struct TripletCentroidNode {
	left: u32,
	right: u32,
	size: u32,
	num_leaves: u32,
	min_leaf: u32,
	max_leaf: u32,
	alive: bool,
	on_heavy_path: bool,
}

impl TripletCentroidNode {
	fn is_leaf(self) -> bool {
		self.left == u32::MAX
	}
}

#[derive(Clone, Copy)]
struct TripletColors {
	red_min: u32,
	red_max: u32,
	blue_min: u32,
	blue_max: u32,
}

impl TripletColors {
	fn is_red(self, leaf: u32) -> bool {
		(self.red_min..=self.red_max).contains(&leaf)
	}

	fn is_blue(self, leaf: u32) -> bool {
		(self.blue_min..=self.blue_max).contains(&leaf)
	}

	fn is_colored(self, leaf: u32) -> bool {
		self.is_red(leaf) || self.is_blue(leaf)
	}
}

#[derive(Clone, Copy)]
struct TripletContractNode {
	leaf: u32,
	same_red: u64,
	red: u64,
}

impl TripletContractNode {
	fn leaf(leaf: u32) -> Self {
		Self {
			leaf,
			same_red: 0,
			red: 0,
		}
	}

	fn internal() -> Self {
		Self {
			leaf: u32::MAX,
			same_red: 0,
			red: 0,
		}
	}

	fn is_leaf(self) -> bool {
		self.leaf != u32::MAX
	}
}

#[derive(Clone, Copy)]
struct TripletLeafCounts {
	red: u64,
	blue: u64,
}

#[derive(Clone, Copy)]
struct TripletPruneState {
	position: u32,
	same_red: u64,
	red: u64,
	kept: bool,
}

impl TripletPruneState {
	fn position(&self) -> usize {
		self.position as usize
	}
}

#[derive(Clone, Copy)]
enum TripletPruneKind {
	Red,
	Uncolored,
}

struct TripletCounter {
	centroids: Vec<TripletCentroidNode>,
	contracts: Vec<TripletContractNode>,
	current_start: usize,
	current_end: usize,
	colors: TripletColors,
	counting: Vec<TripletLeafCounts>,
	keep_states: Vec<bool>,
	prune_states: Vec<TripletPruneState>,
	num_leaves: u32,
}

impl TripletCounter {
	fn new(first: &BinaryTree, second: &BinaryTree) -> Self {
		let (centroids, leaf_order) = Self::centroids(first);
		let contracts =
			second.postorder()
				.map(|node| {
					second.as_leaf(node).map_or_else(
						TripletContractNode::internal,
						|leaf| {
							TripletContractNode::leaf(
							leaf_order[leaf.usize()],
						)
						},
					)
				})
				.collect::<Vec<_>>();
		let current_end = contracts.len();

		Self {
			centroids,
			contracts,
			current_start: 0,
			current_end,
			colors: TripletColors {
				red_min: 0,
				red_max: 0,
				blue_min: 0,
				blue_max: 0,
			},
			counting: Vec::new(),
			keep_states: Vec::new(),
			prune_states: Vec::new(),
			num_leaves: first.num_leaves(),
		}
	}

	fn centroids(
		tree: &BinaryTree,
	) -> (Vec<TripletCentroidNode>, Vec<u32>) {
		let mut subtree_sizes = vec![0; tree.num_nodes() as usize];
		for node in tree.postorder() {
			subtree_sizes[node.usize()] = if tree.is_leaf(node) {
				1
			} else {
				let [left, right] = tree.children_of(
					tree.as_internal(node).unwrap(),
				);
				1 + subtree_sizes[left.usize()]
					+ subtree_sizes[right.usize()]
			};
		}

		let mut centroids =
			Vec::with_capacity(tree.num_nodes() as usize);
		let mut leaf_order = vec![0; tree.num_leaves() as usize];
		let mut next_leaf = 0;
		let mut stack = vec![tree.root().into()];
		while let Some(node) = stack.pop() {
			let index = centroids.len() as u32;
			if let Some(leaf) = tree.as_leaf(node) {
				leaf_order[leaf.usize()] = next_leaf;
				centroids.push(TripletCentroidNode {
					left: u32::MAX,
					right: u32::MAX,
					size: 1,
					num_leaves: 1,
					min_leaf: next_leaf,
					max_leaf: next_leaf,
					alive: true,
					on_heavy_path: false,
				});
				next_leaf += 1;
				continue;
			}

			let [mut left, mut right] = tree
				.children_of(tree.as_internal(node).unwrap());
			if subtree_sizes[left.usize()]
				< subtree_sizes[right.usize()]
			{
				std::mem::swap(&mut left, &mut right);
			}
			centroids.push(TripletCentroidNode {
				left: index + 1,
				right: index + 1 + subtree_sizes[left.usize()],
				size: subtree_sizes[node.usize()],
				num_leaves: 0,
				min_leaf: 0,
				max_leaf: 0,
				alive: true,
				on_heavy_path: false,
			});
			stack.push(right);
			stack.push(left);
		}

		for index in (0..centroids.len()).rev() {
			let node = centroids[index];
			if node.is_leaf() {
				continue;
			}
			let left = centroids[node.left as usize];
			let right = centroids[node.right as usize];
			centroids[index].num_leaves =
				left.num_leaves + right.num_leaves;
			centroids[index].min_leaf = left.min_leaf;
			centroids[index].max_leaf = right.max_leaf;
		}

		(centroids, leaf_order)
	}

	fn distance(mut self) -> u128 {
		let shared = self.count_component(0, 0);
		choose3(self.num_leaves) - shared
	}

	fn find_centroid(&mut self, root: u32, excluded_size: u32) -> u32 {
		let threshold = u64::from(self.centroids[root as usize].size)
			+ u64::from(excluded_size);
		let mut index = root;
		loop {
			let node = self.centroids[index as usize];
			if u64::from(node.size) * 2 < threshold {
				return index - 1;
			}
			self.centroids[index as usize].on_heavy_path = true;
			if node.is_leaf() {
				return index;
			}
			index += 1;
		}
	}

	fn count_component(&mut self, root: u32, excluded_size: u32) -> u128 {
		let centroid = self.find_centroid(root, excluded_size);
		let node = self.centroids[centroid as usize];
		self.centroids[centroid as usize].alive = false;
		if node.is_leaf() {
			return 0;
		}

		let left = self.centroids[node.left as usize];
		let right = self.centroids[node.right as usize];
		let colors = TripletColors {
			red_min: left.min_leaf,
			red_max: left.max_leaf,
			blue_min: right.min_leaf,
			blue_max: right.max_leaf,
		};
		self.colors = colors;
		let mut shared = self.count_shared();
		let component_start = self.current_start;
		let component_end = self.current_end;

		let left_is_complete = !left.on_heavy_path;
		if left.alive && (!left_is_complete || left.num_leaves >= 3) {
			self.contract_pruned(TripletPruneKind::Red);
			shared +=
				self.count_component(node.left, excluded_size);
			self.restore_component(
				component_start,
				component_end,
				colors,
			);
		}

		if right.alive && right.num_leaves >= 3 {
			self.contract_blue();
			shared += self.count_component(node.right, 0);
			self.restore_component(
				component_start,
				component_end,
				colors,
			);
		}

		if centroid != root {
			self.contract_pruned(TripletPruneKind::Uncolored);
			shared += self.count_component(root, node.size);
			self.restore_component(
				component_start,
				component_end,
				colors,
			);
		}

		shared
	}

	fn restore_component(
		&mut self,
		start: usize,
		end: usize,
		colors: TripletColors,
	) {
		self.contracts.truncate(end);
		self.current_start = start;
		self.current_end = end;
		self.colors = colors;
	}

	fn count_shared(&mut self) -> u128 {
		self.counting.clear();
		let mut shared = 0;
		for index in self.current_start..self.current_end {
			let node = self.contracts[index];
			if node.is_leaf() {
				let blue = u64::from(
					self.colors.is_blue(node.leaf),
				);
				if blue == 1 {
					shared += u128::from(node.same_red);
				}
				self.counting.push(TripletLeafCounts {
					red: u64::from(
						self.colors.is_red(node.leaf),
					) + node.red,
					blue,
				});
				continue;
			}

			let right = self.counting.pop().unwrap();
			let left = self.counting.pop().unwrap();
			let blue = left.blue + right.blue;
			shared += u128::from(choose2_u64(left.red))
				* u128::from(right.blue) + u128::from(
				choose2_u64(left.blue),
			) * u128::from(
				right.red,
			) + u128::from(left.red)
				* u128::from(choose2_u64(right.blue))
				+ u128::from(left.blue)
					* u128::from(choose2_u64(right.red))
				+ u128::from(node.same_red) * u128::from(blue)
				+ u128::from(node.red)
					* u128::from(choose2_u64(blue));
			self.counting.push(TripletLeafCounts {
				red: left.red + right.red + node.red,
				blue,
			});
		}
		debug_assert_eq!(self.counting.len(), 1);
		shared
	}

	fn contract_blue(&mut self) {
		self.keep_states.clear();
		let new_start = self.contracts.len();
		for index in self.current_start..self.current_end {
			let node = self.contracts[index];
			if node.is_leaf() {
				let keep = self.colors.is_blue(node.leaf);
				if keep {
					self.contracts.push(
						TripletContractNode::leaf(
							node.leaf,
						),
					);
				}
				self.keep_states.push(keep);
				continue;
			}

			let right = self.keep_states.pop().unwrap();
			let left = self.keep_states.pop().unwrap();
			if left && right {
				self.contracts
					.push(TripletContractNode::internal());
			}
			self.keep_states.push(left || right);
		}
		debug_assert_eq!(self.keep_states, [true]);
		self.current_start = new_start;
		self.current_end = self.contracts.len();
	}

	fn contract_pruned(&mut self, kind: TripletPruneKind) {
		let new_start = self.contracts.len();
		let mut state_len = 0;
		for index in self.current_start..self.current_end {
			let node = self.contracts[index];
			if node.is_leaf() {
				let keep = match kind {
					TripletPruneKind::Red => {
						self.colors.is_red(node.leaf)
					}
					TripletPruneKind::Uncolored => !self
						.colors
						.is_colored(node.leaf),
				};
				let state = if keep {
					let position = u32::try_from(
						self.contracts.len()
							- new_start,
					)
					.unwrap();
					self.contracts.push(node);
					TripletPruneState {
						position,
						same_red: node.same_red,
						red: node.red,
						kept: true,
					}
				} else {
					let red = node.red
						+ u64::from(matches!(
							kind,
							TripletPruneKind::Uncolored
						));
					TripletPruneState {
						position: 0,
						same_red: choose2_u64(red),
						red,
						kept: false,
					}
				};
				if state_len == self.prune_states.len() {
					self.prune_states.push(state);
				} else {
					self.prune_states[state_len] = state;
				}
				state_len += 1;
				continue;
			}

			let right = state_len - 1;
			let left = right - 1;
			let left_kept = self.prune_states[left].kept;
			let right_kept = self.prune_states[right].kept;
			let state = match (left_kept, right_kept) {
				(false, false) => {
					let red = self.prune_states[left].red
						+ self.prune_states[right].red
						+ node.red;
					TripletPruneState {
						position: 0,
						same_red: choose2_u64(red),
						red,
						kept: false,
					}
				}
				(true, false) | (false, true) => {
					let kept = if left_kept {
						left
					} else {
						right
					};
					let red = self.prune_states[left].red
						+ self.prune_states[right].red
						+ node.red;
					let same_red = self.prune_states[left]
						.same_red + self
						.prune_states[right]
						.same_red + node
						.same_red;
					let contract_index = new_start
						+ self.prune_states[kept]
							.position();
					let position = self.prune_states[kept]
						.position;
					let contract = &mut self.contracts
						[contract_index];
					contract.same_red = same_red;
					contract.red = red;
					TripletPruneState {
						position,
						same_red,
						red,
						kept: true,
					}
				}
				(true, true) => {
					let position = u32::try_from(
						self.contracts.len()
							- new_start,
					)
					.unwrap();
					self.contracts.push(node);
					TripletPruneState {
						position,
						same_red: node.same_red,
						red: node.red,
						kept: true,
					}
				}
			};
			self.prune_states[left] = state;
			state_len -= 1;
		}
		debug_assert!(state_len == 1 && self.prune_states[0].kept);
		self.prune_states.truncate(state_len);
		self.current_start = new_start;
		self.current_end = self.contracts.len();
	}
}

fn choose2_u64(value: u64) -> u64 {
	value * value.saturating_sub(1) / 2
}

fn choose3(value: u32) -> u128 {
	let value = u128::from(value);
	value * value.saturating_sub(1) * value.saturating_sub(2) / 6
}
