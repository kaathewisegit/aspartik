use super::{LikelihoodTrait, Row, Transition};
use skvec::SkVec;

pub struct CpuLikelihood<const N: usize> {
	probabilities: Vec<SkVec<Row<N>>>,

	updated_nodes: Vec<usize>,
}

impl<const N: usize> LikelihoodTrait<N> for CpuLikelihood<N> {
	fn propose(
		&mut self,
		nodes: &[usize],
		transitions: &[Transition<N>],
		children: &[usize],
	) -> f64 {
		assert_eq!(nodes.len() * 2, transitions.len());
		assert_eq!(nodes.len() * 2, children.len());

		self.updated_nodes = nodes.to_vec();

		for probability in &mut self.probabilities {
			for i in 0..nodes.len() {
				let left = transitions[i * 2]
					* probability[children[i * 2]];
				let right = transitions[i * 2 + 1]
					* probability[children[i * 2 + 1]];
				probability.set(nodes[i], left * right);
			}
		}

		let root = *nodes.last().unwrap();
		self.probabilities.iter().map(|p| p[root].sum().ln()).sum()
	}

	fn accept(&mut self) {
		for probability in &mut self.probabilities {
			probability.accept();
		}
	}

	fn reject(&mut self) {
		let nodes = std::mem::take(&mut self.updated_nodes);

		for probability in &mut self.probabilities {
			for node in &nodes {
				probability.reject_element(*node);
			}
			// All of the edited items have been manually unset, so
			// there's no need for `accept` or `reject`.
		}
	}
}

impl<const N: usize> CpuLikelihood<N> {
	pub fn new(sites: Vec<Vec<Row<N>>>) -> Self {
		let mut probabilities = vec![
			SkVec::repeat(
				Row::<N>::default(),
				sites[0].len() * 2 - 1
			);
			sites.len()
		];

		for (i, site) in sites.iter().enumerate() {
			for (j, row) in site.iter().enumerate() {
				probabilities[i].set(j, *row);
			}
		}

		for (rows, probability) in sites.iter().zip(&mut probabilities)
		{
			for (i, row) in rows.iter().enumerate() {
				probability.set(i, *row);
			}
			probability.accept();
		}

		Self {
			probabilities,
			updated_nodes: Vec::new(),
		}
	}
}
