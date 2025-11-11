use anyhow::{Context, Result, anyhow, bail};
use cudarc::{
	driver::{
		CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
		PushKernelArg, sys::is_culib_present,
	},
	nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use data::{DnaNucleotide, Msa};

use std::sync::Arc;

use super::{LikelihoodTrait, Row, Transition};
use crate::{tree::Internal, util::msa_to_likelihoods};

const CUDA_SRC: &str =
	concat!(include_str!("typedefs.h"), include_str!("kernels.cu"),);

pub struct CudaLikelihood {
	stream: Arc<CudaStream>,

	propose_fn: CudaFunction,
	copy_projections_fn: CudaFunction,
	update_leaves_fn: CudaFunction,
	update_likelihoods_fn: CudaFunction,

	/// Leaf likelihoods
	///
	/// Has the size of `num_sites * num_leaves`.
	///
	/// Values are grouped by leaves.  The slice is divided into rows of the
	/// length `num_sites`.  `0` to `num_sites` is leaf 0, `num_sites` to `2
	/// * num_sites` is leaf 1, and so on.
	leaves: CudaSlice<Row<4>>,

	/// A contiguous array which stores projection likelihoods
	///
	/// It has the length of `num_edges * num_sites`, with each likelihood
	/// being associated with a particular edge.  Stored in the same way as
	/// `leaves`, except grouped by edges.
	projections: CudaSlice<Row<4>>,

	/// A copy of `projections`
	projections_backup: CudaSlice<Row<4>>,

	/// `num_sites`-long root likelihoods
	likelihoods: CudaSlice<f64>,

	/// Host storage for `likelihoods` to avoid repeating allocations
	host_likelihoods: Vec<f64>,

	/// Edges updated in the current proposal
	///
	/// For each updated node this is the index of the edge leading to its
	/// parent.  Has the length of `num_updated_nodes`.
	edges: CudaSlice<u32>,

	/// Transitions for each edge from `edges`
	///
	/// The length is `num_updated_nodes`.
	transitions: CudaSlice<Transition<4>>,

	/// Nodes updated in the current proposal
	///
	/// The length is `num_updated_nodes`.
	nodes: CudaSlice<u32>,

	scales: CudaSlice<u8>,
	scales_backup: CudaSlice<u8>,
	scale_sums: CudaSlice<u32>,
	scale_sums_backup: Vec<u32>,

	/// Total number of sites
	///
	/// Immutable, passed to the kernels.
	num_sites: u32,

	/// Total number of leaves
	///
	/// Immutable, passed to the kernels.
	num_leaves: u32,

	/// Number of nodes updated in the current proposal
	///
	/// Changes on each step.
	num_updated_nodes: u32,
}

impl LikelihoodTrait<4> for CudaLikelihood {
	type Arguments = (usize,);

	fn new(
		msa: Msa<DnaNucleotide>,
		(cuda_device,): Self::Arguments,
	) -> Result<Self> {
		// SAFETY: since the function tries to link the library [0], it
		// could theoretically be unsafe.  In practice, the resulting
		// library is discarded.
		//
		// [0]: https://docs.rs/cudarc/0.17.3/src/cudarc/driver/sys/mod.rs.html#26256
		let is_cuda_enabled = unsafe { is_culib_present() };
		if !is_cuda_enabled {
			bail!("CUDA library not found");
		}

		let num_sites = msa.num_sites();
		let num_leaves = msa.num_sequences();
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let context = CudaContext::new(cuda_device)?;
		let stream = context.new_stream()?;

		// SAFETY: `CudaLikelihood` only uses a single stream, so
		// there's no need for cross-stream synchronization
		unsafe { context.disable_event_tracking() };

		let leaves = stream.memcpy_stod(&msa_to_likelihoods(msa))?;
		let projections: CudaSlice<Row<4>> =
			stream.alloc_zeros(num_edges * num_sites)?;
		let projections_backup: CudaSlice<Row<4>> =
			stream.alloc_zeros(num_edges * num_sites)?;

		let likelihoods: CudaSlice<f64> =
			stream.alloc_zeros(num_sites)?;
		let edges = stream.alloc_zeros(num_edges)?;
		let transitions = stream.alloc_zeros(num_edges)?;
		let nodes = stream.alloc_zeros(num_nodes)?;

		let scales = stream.alloc_zeros(num_edges * num_sites)?;
		let scales_backup =
			stream.alloc_zeros(num_edges * num_sites)?;
		let scale_sums = stream.alloc_zeros(num_sites)?;
		let scale_sums_backup = vec![0; num_sites];

		let opts = CompileOptions {
			include_paths: vec![
				"/usr/local/cuda/include/".to_owned()
			],
			..Default::default()
		};
		let ptx = compile_ptx_with_opts(CUDA_SRC, opts)?;

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

			scales,
			scales_backup,
			scale_sums,
			scale_sums_backup,

			num_sites: num_sites as u32,
			num_leaves: num_leaves as u32,

			num_updated_nodes: 0,
		})
	}

	/// Propose an edit to the tree
	///
	/// Asynchronous.  This method starts the GPU calculations and returns
	/// right after.  The job synchronization is handled by the [shared
	/// stream][Self::stream].
	fn propose(
		&mut self,
		nodes: &[usize],
		edges: &[usize],
		transitions: &[Transition<4>],
		leaves_end: usize,
		root: usize,
		frequencies: Row<4>,
	) -> Result<()> {
		self.num_updated_nodes = nodes.len() as u32;
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		let nodes: Vec<_> = nodes.iter().map(|n| *n as u32).collect();
		let edges: Vec<_> = edges.iter().map(|e| *e as u32).collect();

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

		self.update_likelihoods(root as u32, frequencies)?;

		Ok(())
	}

	/// Fetches the likelihoods calculated by [`update_likelihoods`]
	///
	/// Synchronous, blocks on the `self.likelihoods` buffer.
	///
	/// [`update_likelihoods`]: Self::update_likelihoods
	fn likelihood(&mut self, weights: &[f64]) -> Result<f64> {
		self.stream.memcpy_dtoh(
			&self.likelihoods,
			&mut self.host_likelihoods,
		)?;

		let mut out: f64 = 0.0;

		for (likelihood, weight) in
			self.host_likelihoods.iter().zip(weights)
		{
			out += likelihood * weight;
		}

		let scale_sums = self.stream.memcpy_dtov(&self.scale_sums)?;
		for (scale, weight) in scale_sums.iter().zip(weights) {
			out -= f64::from(*scale) * weight;
		}

		Ok(out)
	}

	fn accept(&mut self) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		self.stream.memcpy_dtoh(
			&self.scale_sums,
			&mut self.scale_sums_backup,
		)?;

		self.copy_projections(true)?;

		self.num_updated_nodes = 0;
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		self.stream.memcpy_htod(
			&self.scale_sums_backup,
			&mut self.scale_sums,
		)?;

		self.copy_projections(false)?;

		self.num_updated_nodes = 0;
		Ok(())
	}
}

impl CudaLikelihood {
	/// Updates both leaves and internal nodes
	///
	/// Asynchronous.
	fn update_all(
		&self,
		leaves_end: u32,
		internals_start: u32,
	) -> Result<()> {
		let mut builder = self.stream.launch_builder(&self.propose_fn);

		let block_size = 16 * 4;
		let num_site_blocks = (self.num_sites * 4).div_ceil(block_size);
		let cfg = LaunchConfig {
			grid_dim: (num_site_blocks, 1, 1),
			block_dim: (block_size, 1, 1),
			shared_mem_bytes: 0,
		};

		builder.arg(&self.num_sites);
		builder.arg(&self.num_leaves);

		builder.arg(&self.leaves);
		builder.arg(&self.projections);
		builder.arg(&self.scales);
		builder.arg(&self.scale_sums);

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

	/// Update leaf-originating partial likelihoods
	///
	/// Asynchronous.
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

	/// Calculates the root node likelihood from its two projections
	///
	/// Asynchronous.
	fn update_likelihoods(
		&self,
		root: u32,
		frequencies: Row<4>,
	) -> Result<()> {
		let mut builder =
			self.stream.launch_builder(&self.update_likelihoods_fn);

		let cfg = self.cfg(32, 1);

		builder.arg(&self.num_sites);
		builder.arg(&self.num_leaves);

		builder.arg(&self.projections);
		builder.arg(&self.likelihoods);

		builder.arg(&self.edges);

		builder.arg(&root);
		builder.arg(&frequencies);

		// SAFETY: TODO
		unsafe { builder.launch(cfg) }.with_context(|| {
			anyhow!("update_likelihoods: {cfg:?}")
		})?;

		Ok(())
	}

	/// Applies a copy function to all of the updated edges.
	///
	/// This is an abstraction which unifies `accept` and `reject`, since
	/// they are basically the same.
	///
	/// Asynchronous.
	fn copy_projections(&mut self, accept: bool) -> Result<()> {
		let cfg = self.cfg(128, self.num_updated_nodes);

		let mut builder =
			self.stream.launch_builder(&self.copy_projections_fn);

		builder.arg(&self.num_sites);

		if accept {
			builder.arg(&self.projections);
			builder.arg(&self.projections_backup);

			builder.arg(&self.scales);
			builder.arg(&self.scales_backup);
		} else {
			builder.arg(&self.projections_backup);
			builder.arg(&self.projections);

			builder.arg(&self.scales_backup);
			builder.arg(&self.scales);
		}

		builder.arg(&self.edges);

		// SAFETY: TODO
		unsafe { builder.launch(cfg) }.with_context(|| {
			anyhow!("copy_projections({accept}): {cfg:#?}")
		})?;

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

	pub fn pattern_likelihoods(
		&self,
		root: Internal,
		frequencies: Row<4>,
	) -> Result<Vec<f64>> {
		self.update_likelihoods(root.index() as u32, frequencies)?;

		let likelihoods = self.stream.memcpy_dtov(&self.likelihoods)?;

		Ok(likelihoods)
	}
}
