use anyhow::{anyhow, bail, Context, Result};
use cudarc::{
	driver::{
		CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
		PushKernelArg,
	},
	nvrtc::Ptx,
};

use std::sync::Arc;

use super::{LikelihoodTrait, Row, Transition};
use crate::util::transpose;

/// This is somewhat of a hack.  The PTX assembly file is compiled in build.rs
/// and the variable contains the path to it.  However, I didn't gate off this
/// module behind the CUDA features because rust-analyzer would start warning
/// about CUDA variables in super used here being unused.  So, instead the `new`
/// constructor bails if the CUDA feature isn't enabled, so we never see the
/// dummy empty file which is created when the CUDA feature isn't enabled.
const PTX_SRC: &str = include_str!(env!("ASPARTIK_B3_PTX_SRC_PATH"));

pub struct CudaLikelihood {
	stream: Arc<CudaStream>,

	propose_fn: CudaFunction,
	copy_projections_fn: CudaFunction,
	update_leaves_fn: CudaFunction,
	update_likelihoods_fn: CudaFunction,

	leaves: CudaSlice<Row<4>>,
	projections: CudaSlice<Row<4>>,
	projections_backup: CudaSlice<Row<4>>,
	likelihoods: CudaSlice<f64>,
	host_likelihoods: Vec<f64>,
	edges: CudaSlice<u32>,
	transitions: CudaSlice<Transition<4>>,
	nodes: CudaSlice<u32>,

	num_sites: u32,
	num_leaves: u32,
	num_updated_nodes: u32,
}

impl LikelihoodTrait<4> for CudaLikelihood {
	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<4>],
		leaves_end: usize,
		root: usize,
	) -> Result<()> {
		let nodes: Vec<_> = nodes.iter().map(|n| *n as u32).collect();
		let edges: Vec<_> = edges.iter().map(|e| *e as u32).collect();

		self.num_updated_nodes = nodes.len() as u32;

		self.stream.memcpy_htod(&edges, &mut self.edges)?;
		self.stream.memcpy_htod(&nodes, &mut self.nodes)?;
		self.stream
			.memcpy_htod(transitions, &mut self.transitions)?;

		let mut leaves_end = leaves_end as u32;
		let internals_start = leaves_end;

		if leaves_end > 10 {
			self.update_leaves(leaves_end)?;
			leaves_end = 0;
		}
		self.update_all(leaves_end, internals_start)?;

		self.update_likelihoods(root as u32)?;

		Ok(())
	}

	fn likelihood(&mut self) -> Result<f64> {
		self.stream.memcpy_dtoh(
			&self.likelihoods,
			&mut self.host_likelihoods,
		)?;

		Ok(self.host_likelihoods.iter().sum())
	}

	fn accept(&mut self) -> Result<()> {
		self.copy_projections(true)
	}

	fn reject(&mut self) -> Result<()> {
		self.copy_projections(false)
	}
}

impl CudaLikelihood {
	fn update_all(
		&self,
		leaves_end: u32,
		internals_start: u32,
	) -> Result<()> {
		let mut builder = self.stream.launch_builder(&self.propose_fn);

		let block_size = 16;
		let num_site_blocks = self.num_sites.div_ceil(block_size);
		let cfg = LaunchConfig {
			grid_dim: (num_site_blocks, 1, 1),
			block_dim: (block_size, 4, 1),
			shared_mem_bytes: 0,
		};

		builder.arg(&self.num_sites);
		builder.arg(&self.num_leaves);

		builder.arg(&self.leaves);
		builder.arg(&self.projections);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.nodes);
		builder.arg(&self.edges);
		builder.arg(&self.transitions);

		builder.arg(&leaves_end);
		builder.arg(&internals_start);

		// TODO: safety
		unsafe { builder.launch(cfg) }
			.with_context(|| anyhow!("update_all: {cfg:?}"))?;

		Ok(())
	}

	fn update_leaves(&self, leaves_end: u32) -> Result<()> {
		let mut builder =
			self.stream.launch_builder(&self.update_leaves_fn);

		let block_size = 16;
		let num_site_blocks = self.num_sites.div_ceil(block_size);
		let cfg = LaunchConfig {
			grid_dim: (num_site_blocks, leaves_end, 1),
			block_dim: (block_size, 4, 1),
			shared_mem_bytes: 0,
		};

		builder.arg(&self.num_sites);

		builder.arg(&self.leaves);
		builder.arg(&self.projections);

		builder.arg(&self.nodes);
		builder.arg(&self.edges);
		builder.arg(&self.transitions);

		// TODO: safety
		unsafe { builder.launch(cfg) }
			.with_context(|| anyhow!("update_leaves: {cfg:?}"))?;

		Ok(())
	}

	fn update_likelihoods(&self, root: u32) -> Result<()> {
		let mut builder =
			self.stream.launch_builder(&self.update_likelihoods_fn);

		let cfg = self.cfg(32, 1);

		builder.arg(&self.num_sites);
		builder.arg(&self.num_leaves);

		builder.arg(&self.projections);
		builder.arg(&self.likelihoods);

		builder.arg(&root);

		// TODO: safety
		unsafe { builder.launch(cfg) }.with_context(|| {
			anyhow!("update_likelihoods: {cfg:?}")
		})?;

		Ok(())
	}

	/// Applies a copy function to all of the updated edges.
	///
	/// This is an abstraction which unifies `accept` and `reject`, since
	/// they are basically the same.
	fn copy_projections(&mut self, accept: bool) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		let cfg = self.cfg(128, self.num_updated_nodes);

		let mut builder =
			self.stream.launch_builder(&self.copy_projections_fn);

		builder.arg(&self.num_sites);

		if accept {
			builder.arg(&self.projections);
			builder.arg(&self.projections_backup);
		} else {
			builder.arg(&self.projections_backup);
			builder.arg(&self.projections);
		}

		builder.arg(&self.edges);

		// TODO: safety
		unsafe { builder.launch(cfg) }.with_context(|| {
			let op = if accept { "accept" } else { "reject" };
			anyhow!("{op}: {cfg:#?}")
		})?;

		self.num_updated_nodes = 0;

		Ok(())
	}

	fn cfg(&self, block_size: u32, dim2: u32) -> LaunchConfig {
		let num_site_blocks = self.num_sites.div_ceil(block_size);
		LaunchConfig {
			grid_dim: (num_site_blocks, dim2, 1),
			block_dim: (block_size, 1, 1),
			shared_mem_bytes: 0,
		}
	}

	pub fn new(
		leaves: Vec<Vec<Row<4>>>,
		cuda_device: usize,
	) -> Result<Self> {
		if cfg!(not(feature = "cuda")) {
			bail!("b3 was built without CUDA support");
		}

		let num_sites = leaves.len();
		let num_leaves = leaves[0].len();
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let context = CudaContext::new(cuda_device)?;
		let stream = context.new_stream()?;

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
		let edges = stream.alloc_zeros(num_edges)?;
		let transitions = stream.alloc_zeros(num_edges)?;
		let nodes = stream.alloc_zeros(num_nodes)?;

		let ptx = Ptx::from_src(PTX_SRC);
		let module = context.load_module(ptx)?;
		let propose_fn = module.load_function("propose")?;
		let copy_projections_fn =
			module.load_function("copy_projections")?;
		let update_leaves_fn = module.load_function("update_leaves")?;
		let update_likelihoods_fn =
			module.load_function("update_likelihoods")?;

		Ok(Self {
			stream,

			propose_fn,
			copy_projections_fn,
			update_leaves_fn,
			update_likelihoods_fn,

			leaves,
			projections,
			projections_backup,
			likelihoods,
			host_likelihoods: vec![0.0; num_sites],
			edges,
			transitions,
			nodes,

			num_sites: num_sites as u32,
			num_leaves: num_leaves as u32,

			num_updated_nodes: 0,
		})
	}
}
