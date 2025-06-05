use anyhow::Result;
use cudarc::{
	driver::{
		CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
		PushKernelArg,
	},
	nvrtc::compile_ptx,
};

use std::sync::Arc;

use super::{LikelihoodTrait, Row, Transition};
use crate::util::transpose;

pub struct CudaLikelihood {
	stream: Arc<CudaStream>,
	cfg: LaunchConfig,

	propose_fn: CudaFunction,
	accept_fn: CudaFunction,
	reject_fn: CudaFunction,

	leaves: CudaSlice<Row<4>>,
	projections: CudaSlice<Row<4>>,
	projections_backup: CudaSlice<Row<4>>,
	likelihoods: CudaSlice<f64>,
	host_likelihoods: Vec<f64>,
	updated_edges: CudaSlice<u32>,
	transitions: CudaSlice<Transition<4>>,
	nodes: CudaSlice<u32>,

	num_sites: u32,
	num_leaves: u32,
	num_updated_nodes: u32,
}

const CUDA_MODULE: &str = include_str!("kernels.cu");

impl LikelihoodTrait<4> for CudaLikelihood {
	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<4>],
		cutoff: usize,
		root: usize,
	) -> Result<f64> {
		let nodes: Vec<_> = nodes.iter().map(|n| *n as u32).collect();
		let edges: Vec<_> = edges.iter().map(|e| *e as u32).collect();

		self.num_updated_nodes = nodes.len() as u32;

		self.stream.memcpy_htod(&edges, &mut self.updated_edges)?;
		self.stream.memcpy_htod(&nodes, &mut self.nodes)?;
		self.stream
			.memcpy_htod(transitions, &mut self.transitions)?;

		let cutoff = cutoff as u32;
		let root = root as u32;

		let mut builder = self.stream.launch_builder(&self.propose_fn);

		builder.arg(&self.num_sites);
		builder.arg(&self.num_leaves);

		builder.arg(&self.leaves);
		builder.arg(&self.projections);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.nodes);
		builder.arg(&self.updated_edges);
		builder.arg(&self.transitions);
		builder.arg(&cutoff);
		builder.arg(&root);

		builder.arg(&self.likelihoods);

		// TODO: safety
		unsafe { builder.launch(self.cfg) }?;

		self.stream.memcpy_dtoh(
			&self.likelihoods,
			&mut self.host_likelihoods,
		)?;

		Ok(self.host_likelihoods.iter().sum())
	}

	fn accept(&mut self) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		let mut builder = self.stream.launch_builder(&self.accept_fn);

		builder.arg(&self.num_sites);

		builder.arg(&self.projections);
		builder.arg(&self.projections_backup);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.updated_edges);

		// TODO: safety
		unsafe { builder.launch(self.cfg) }?;

		self.num_updated_nodes = 0;

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		let mut builder = self.stream.launch_builder(&self.reject_fn);

		builder.arg(&self.num_sites);

		builder.arg(&self.projections);
		builder.arg(&self.projections_backup);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.updated_edges);

		// TODO: safety
		unsafe { builder.launch(self.cfg) }?;

		self.num_updated_nodes = 0;

		Ok(())
	}
}

impl CudaLikelihood {
	pub fn new(
		leaves: Vec<Vec<Row<4>>>,
		cuda_device: usize,
	) -> Result<Self> {
		let num_sites = leaves.len();
		let num_leaves = leaves[0].len();
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let context = CudaContext::new(cuda_device)?;
		let stream = context.default_stream();

		// SAFETY: CudaLikelihood only uses a single stream, so there's
		// no need for cross-stream synchronization
		unsafe { context.disable_event_tracking() };

		let leaves = stream.memcpy_stod(&transpose(leaves))?;
		let projections: CudaSlice<Row<4>> =
			stream.alloc_zeros(num_edges * num_sites)?;
		let projections_backup: CudaSlice<Row<4>> =
			stream.alloc_zeros(num_edges * num_sites)?;

		let likelihoods: CudaSlice<f64> =
			stream.alloc_zeros(num_sites)?;
		let updated_edges = stream.alloc_zeros(num_edges)?;
		let transitions = stream.alloc_zeros(num_edges)?;
		let nodes = stream.alloc_zeros(num_nodes)?;

		let ptx = compile_ptx(CUDA_MODULE)?;
		let module = context.load_module(ptx)?;
		let propose_fn = module.load_function("propose")?;
		let accept_fn = module.load_function("accept")?;
		let reject_fn = module.load_function("reject")?;

		const SIZE: u32 = 32;
		let cfg = LaunchConfig {
			grid_dim: ((num_sites as u32).div_ceil(SIZE), 1, 1),
			block_dim: (SIZE, 1, 1),
			shared_mem_bytes: 0,
		};

		Ok(Self {
			stream,
			cfg,

			propose_fn,
			accept_fn,
			reject_fn,

			leaves,
			projections,
			projections_backup,
			likelihoods,
			host_likelihoods: vec![0.0; num_sites],
			updated_edges,
			transitions,
			nodes,

			num_sites: num_sites as u32,
			num_leaves: num_leaves as u32,

			num_updated_nodes: 0,
		})
	}
}
