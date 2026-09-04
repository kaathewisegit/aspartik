use anyhow::{Result, ensure};
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::BinaryTree;

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
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

pub fn robinson_foulds_matrix(trees: &[BinaryTree]) -> Result<Vec<Vec<u32>>> {
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
	}

	let num_clades = usize::try_from(num_leaves - 2)?;
	let capacity = trees.len().checked_mul(num_clades).unwrap_or(0);
	let mut clades =
		FxHashMap::<CladeHash, Vec<usize>>::with_capacity_and_hasher(
			capacity,
			FxBuildHasher,
		);

	for (tree_index, tree) in trees.iter().enumerate() {
		let mut hashes =
			vec![CladeHash::default(); tree.num_nodes() as usize];
		for node in tree.postorder() {
			let hash = if let Some(leaf) = tree.as_leaf(node) {
				CladeHash::leaf(leaf.index())
			} else {
				let internal = tree.as_internal(node).unwrap();
				let (left, right) = tree.children_of(internal);
				hashes[left.index() as usize]
					.combine(hashes[right.index() as usize])
			};
			hashes[node.index() as usize] = hash;

			if node != tree.root().into() && tree.is_internal(node)
			{
				let tree_indices =
					clades.entry(hash).or_default();
				if tree_indices.last() != Some(&tree_index) {
					tree_indices.push(tree_index);
				}
			}
		}
	}

	let max_distance = (num_leaves - 2) * 2;
	let mut distances = vec![vec![max_distance; trees.len()]; trees.len()];
	for (index, row) in distances.iter_mut().enumerate() {
		row[index] = 0;
	}

	for tree_indices in clades.values() {
		for (offset, &first) in tree_indices.iter().enumerate() {
			for &second in &tree_indices[offset + 1..] {
				distances[first][second] -= 2;
				distances[second][first] -= 2;
			}
		}
	}

	Ok(distances)
}
