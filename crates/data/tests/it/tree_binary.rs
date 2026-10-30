use arbitrary::{Arbitrary, Unstructured};
use arbtest::arbtest;
use rand::SeedableRng;
use rand_pcg::Pcg64;

use data::{TaxonSet, tree::BinaryTree};

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
