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

pub struct CudaLikelihood {
	stream: Arc<CudaStream>,
	cfg: LaunchConfig,

	propose_fn: CudaFunction,
	reject_fn: CudaFunction,

	leaves: CudaSlice<Row<4>>,
	projections: CudaSlice<Row<4>>,
	masks: CudaSlice<u8>,
	likelihoods: CudaSlice<f64>,
	updated_edges: CudaSlice<u32>,

	num_edges: u32,
	num_sites: u32,
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
		let nodes = self.stream.memcpy_stod(&nodes)?;
		self.stream.memcpy_htod(&edges, &mut self.updated_edges)?;
		let transitions = self.stream.memcpy_stod(transitions)?;
		let cutoff = cutoff as u32;
		let root = root as u32;

		let mut builder = self.stream.launch_builder(&self.propose_fn);

		builder.arg(&self.num_edges);
		builder.arg(&self.num_sites);
		builder.arg(&self.leaves);
		builder.arg(&self.masks);
		builder.arg(&self.projections);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&nodes);
		builder.arg(&self.updated_edges);
		builder.arg(&transitions);
		builder.arg(&cutoff);
		builder.arg(&root);

		builder.arg(&self.likelihoods);

		// TODO: safety
		let events = unsafe { builder.launch(self.cfg) }?;
		if let Some((left, right)) = events {
			self.stream.wait(&left)?;
			self.stream.wait(&right)?;
		}

		let likelihoods = self.stream.memcpy_dtov(&self.likelihoods)?;

		Ok(likelihoods.into_iter().map(|l| l.ln()).sum())
	}

	fn accept(&mut self) -> Result<()> {
		self.num_updated_nodes = 0;
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		let mut builder = self.stream.launch_builder(&self.reject_fn);

		builder.arg(&self.num_edges);
		builder.arg(&self.num_sites);

		builder.arg(&self.masks);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.updated_edges);

		// TODO: safety
		let events = unsafe { builder.launch(self.cfg) }?;
		if let Some((left, right)) = events {
			self.stream.wait(&left)?;
			self.stream.wait(&right)?;
		}

		self.num_updated_nodes = 0;

		Ok(())
	}
}

impl CudaLikelihood {
	pub fn new(leaves: Vec<Vec<Row<4>>>) -> Result<Self> {
		let num_sites = leaves.len();
		let num_leaves = leaves[0].len();
		let num_internals = num_leaves - 1;
		let num_edges = num_internals * 2;

		let context = CudaContext::new(0)?;
		let stream = context.default_stream();

		let leaves = stream.memcpy_stod(&transpose(leaves))?;
		let projections: CudaSlice<Row<4>> =
			stream.alloc_zeros(num_edges * num_sites * 2)?;
		let masks: CudaSlice<u8> =
			stream.alloc_zeros(num_edges * num_sites)?;

		let likelihoods: CudaSlice<f64> =
			stream.alloc_zeros(num_sites)?;
		let updated_edges = stream.alloc_zeros(num_edges)?;

		let ptx = compile_ptx(CUDA_MODULE)?;
		let module = context.load_module(ptx)?;
		let reject_fn = module.load_function("reject")?;
		let propose_fn = module.load_function("propose")?;

		let cfg = LaunchConfig {
			grid_dim: ((num_sites as u32 + 31) / 32, 1, 1),
			block_dim: (32, 1, 1),
			shared_mem_bytes: 0,
		};

		Ok(Self {
			stream,
			cfg,

			propose_fn,
			reject_fn,

			leaves,
			projections,
			masks,
			likelihoods,
			updated_edges,

			num_updated_nodes: 0,
			num_edges: num_edges as u32,
			num_sites: num_sites as u32,
		})
	}
}

fn transpose(leaves: Vec<Vec<Row<4>>>) -> Vec<Row<4>> {
	let num_sites = leaves.len();
	let num_edges = leaves[0].len();

	let mut out = Vec::with_capacity(num_sites * num_edges);

	for edge in 0..num_edges {
		#[expect(clippy::needless_range_loop)]
		for site in 0..num_sites {
			out.push(leaves[site][edge]);
		}
	}

	out
}
