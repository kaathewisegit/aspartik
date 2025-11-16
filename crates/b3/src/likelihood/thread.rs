use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use num_traits::Zero;

use std::{sync::Arc, thread};

use super::{CpuLikelihood, LikelihoodTrait, Linalg4, Space};
use data::{DnaNucleotide, Msa};

type Update<S> = (
	Vec<usize>,
	Vec<usize>,
	Vec<<S as Space>::Matrix>,
	usize,
	usize,
	<S as Space>::Vector,
);

pub struct ThreadedLikelihood<S: Space> {
	updates: Vec<Sender<Arc<Update<S>>>>,
	likelihoods: Vec<Receiver<S::Scalar>>,
	accepts: Vec<Sender<bool>>,

	/// The MCMC might call `accept` or `reject` without calling `propose`.
	/// In these cases sending accept/rejects will mess things up, as the
	/// threads will be waiting for updates.
	has_proposed: bool,
}

impl<S: Space> LikelihoodTrait for ThreadedLikelihood<S> {
	type S = S;

	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[S::Matrix],
		leaves_end: usize,
		root: usize,
		frequencies: S::Vector,
	) -> Result<()> {
		let update = Arc::new((
			nodes.to_owned(),
			edges.to_owned(),
			transitions.to_owned(),
			leaves_end,
			root,
			frequencies,
		));
		for sender in &self.updates {
			sender.send(update.clone())?;
		}

		self.has_proposed = true;

		Ok(())
	}

	fn likelihood(&mut self) -> Result<S::Scalar> {
		let mut out = S::Scalar::zero();
		for receiver in &self.likelihoods {
			out += receiver.recv()?;
		}

		Ok(out)
	}

	fn accept(&mut self) -> Result<()> {
		if !self.has_proposed {
			return Ok(());
		}
		for sender in &self.accepts {
			sender.send(true)?;
		}
		self.has_proposed = false;
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		if !self.has_proposed {
			return Ok(());
		}
		for sender in &self.accepts {
			sender.send(false)?;
		}
		self.has_proposed = false;
		Ok(())
	}
}

impl ThreadedLikelihood<Linalg4> {
	pub fn new(msa: Msa<DnaNucleotide>, thread_split_size: usize) -> Self {
		let mut update_senders = Vec::new();
		let mut likelihoods_receivers = Vec::new();
		let mut accept_senders = Vec::new();

		let num_sites = msa.num_sites();
		let num_threads = num_sites.div_ceil(thread_split_size);
		let segment_len = num_sites / num_threads;

		for i in 0..num_threads {
			let (update_sender, update_receiver) = bounded(1);
			update_senders.push(update_sender);

			let (likelihood_sender, likelihood_receiver) =
				bounded(1);
			likelihoods_receivers.push(likelihood_receiver);

			let (accept_sender, accept_receiver) = bounded(1);
			accept_senders.push(accept_sender);

			let start = i * segment_len;
			let end = if i == num_threads - 1 {
				num_sites
			} else {
				(i + 1) * segment_len
			};
			let msa_subset = msa.slice(start..end);

			thread::spawn(move || {
				worker(
					msa_subset,
					update_receiver,
					likelihood_sender,
					accept_receiver,
				);
			});
		}

		Self {
			updates: update_senders,
			likelihoods: likelihoods_receivers,
			accepts: accept_senders,

			has_proposed: false,
		}
	}
}

fn worker(
	msa: Msa<DnaNucleotide>,
	update_receiver: Receiver<Arc<Update<Linalg4>>>,
	likelihood_sender: Sender<f64>,
	accept_receiver: Receiver<bool>,
) {
	let mut cpu = CpuLikelihood::new(msa);

	loop {
		let Ok(update) = update_receiver.recv() else {
			// Parent process has been dropped, meaning we should
			// terminate too.
			break;
		};
		cpu.propose(
			&update.0, &update.1, &update.2, update.3, update.4,
			update.5,
		)
		.unwrap();

		let likelihood = cpu.likelihood().unwrap();
		likelihood_sender.send(likelihood).unwrap();

		let accept = accept_receiver.recv().unwrap();
		if accept {
			cpu.accept().unwrap();
		} else {
			cpu.reject().unwrap();
		}
	}
}
