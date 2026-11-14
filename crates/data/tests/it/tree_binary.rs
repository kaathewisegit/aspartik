use arbitrary::{Arbitrary, Unstructured};
use arbtest::arbtest;
use rand::SeedableRng;
use rand_pcg::Pcg64;

use data::{
	TaxonSet,
	tree::{BinaryTree, Node, RootedTree, triplet_distance_matrix},
};

#[test]
fn random_tree() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2..=1_000)?;
		let seed = u64::arbitrary(u)?;
		let mut rng = Pcg64::seed_from_u64(seed);

		let taxa = TaxonSet::ranged_ints(num_leaves);
		let tree = BinaryTree::random(taxa, &mut rng).unwrap();
		tree.validate().unwrap();

		assert_eq!(tree.leaves().count(), num_leaves);
		assert_eq!(
			tree.internals().count(),
			usize::try_from(tree.num_internals()).unwrap()
		);
		for leaf in tree.leaves() {
			assert!(tree.is_leaf(leaf.into()));
			assert!(!tree.is_internal(leaf.into()));
		}
		for internal in tree.internals() {
			assert!(!tree.is_leaf(internal.into()));
			assert!(tree.is_internal(internal.into()));
		}

		Ok(())
	});
}

fn triplet_topology(tree: &BinaryTree, leaves: [Node; 3]) -> usize {
	let [a, b, c] = leaves;
	let ancestors = [tree.mrca(a, b), tree.mrca(a, c), tree.mrca(b, c)];
	let depths = ancestors.map(|mut node| {
		let mut depth = 0;
		while let Some(parent) = tree.parent_of(node) {
			depth += 1;
			node = parent.into();
		}
		depth
	});
	(0..3).max_by_key(|&index| depths[index]).unwrap()
}

fn reference_triplet(first: &BinaryTree, second: &BinaryTree) -> u64 {
	let mut distance = 0;
	for a in 0..first.num_leaves() {
		for b in a + 1..first.num_leaves() {
			for c in b + 1..first.num_leaves() {
				let leaves = [Node(a), Node(b), Node(c)];
				distance += u64::from(
					triplet_topology(first, leaves)
						!= triplet_topology(
							second, leaves,
						),
				);
			}
		}
	}
	distance
}

#[test]
fn triplet_matches_enumeration() {
	arbtest(|u: &mut Unstructured<'_>| {
		let num_leaves = u.int_in_range(2..=25)?;
		let mut rng = Pcg64::seed_from_u64(u64::arbitrary(u)?);
		let taxa = TaxonSet::ranged_ints(num_leaves);
		let first = BinaryTree::random(taxa.clone(), &mut rng).unwrap();
		let second = BinaryTree::random(taxa, &mut rng).unwrap();
		let expected = reference_triplet(&first, &second);
		let actual: u64 = first.triplet_distance(&second);
		assert_eq!(actual, expected);
		assert_eq!(second.triplet_distance(&first), expected);
		assert_eq!(first.triplet_distance(&first), 0);
		Ok(())
	});
}

#[test]
fn triplet_matrix_thread_counts() {
	let mut rng = Pcg64::seed_from_u64(4);
	let taxa = TaxonSet::ranged_ints(20);
	let trees = (0..8)
		.map(|_| BinaryTree::random(taxa.clone(), &mut rng).unwrap())
		.collect::<Vec<_>>();
	let references = trees.iter().collect::<Vec<_>>();
	for threads in [1, 2, 4] {
		let matrix =
			triplet_distance_matrix(&references, Some(threads))
				.unwrap();
		let expected = trees
			.iter()
			.flat_map(|first| {
				trees.iter().map(move |second| {
					reference_triplet(first, second)
				})
			})
			.collect::<Vec<_>>();
		assert_eq!(matrix.as_slice(), expected);
	}
	assert!(triplet_distance_matrix(&references, Some(0)).is_err());
	assert!(triplet_distance_matrix(&[], Some(1))
		.unwrap()
		.as_slice()
		.is_empty());
	assert_eq!(
		triplet_distance_matrix(&references[..1], Some(1))
			.unwrap()
			.as_slice(),
		[0]
	);
}
