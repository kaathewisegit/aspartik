use anyhow::{Context, Result, anyhow, bail};
use cudarc::{
	driver::{
		CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
		PushKernelArg, sys::is_culib_present,
	},
	nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use std::sync::Arc;

use super::Calculator;

type Row = [f64; 4];
type Transition = [f64; 16];

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
	leaves: CudaSlice<u8>,

	/// A contiguous array which stores projection likelihoods
	///
	/// It has the length of `num_nodes * num_sites`, with each likelihood
	/// being associated with a particular edge.  Stored in the same way as
	/// `leaves`, except grouped by edges.
	projections: CudaSlice<Row>,

	/// A copy of `projections`
	projections_backup: CudaSlice<Row>,

	/// `num_sites`-long root likelihoods
	likelihoods: CudaSlice<f64>,

	/// Children of internal nodes updated in current proposal
	children: CudaSlice<u32>,

	/// Transitions for each edge from `edges`
	///
	/// The length is `num_updated_nodes`.
	transitions: CudaSlice<Transition>,

	/// Nodes updated in the current proposal
	///
	/// The length is `num_updated_nodes`.
	nodes: CudaSlice<u32>,

	scales: CudaSlice<u8>,
	scales_backup: CudaSlice<u8>,
	scale_sums: CudaSlice<u32>,
	scale_sums_backup: Vec<u32>,

	/// Total number of sites
	num_sites: u32,

	/// Number of nodes updated in the current proposal
	///
	/// Changes on each step.
	num_updated_nodes: u32,
}

impl Calculator<4, f64> for CudaLikelihood {
	/// Propose an edit to the tree
	///
	/// Asynchronous.  This method starts the GPU calculations and returns
	/// right after.  The job synchronization is handled by the [shared
	/// stream][Self::stream].
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[(usize, usize)],
		transitions: &[[[f64; 4]; 4]],
		leaves_end: usize,
		frequencies: [f64; 4],
	) -> Result<()> {
		self.num_updated_nodes = nodes.len() as u32 - 1;
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		let root_children = children.last().unwrap();
		let nodes: Vec<_> = nodes.iter().map(|n| *n as u32).collect();
		let children: Vec<_> = children
			.iter()
			.flat_map(|&(l, r)| [l as u32, r as u32])
			.collect();

		self.stream.memcpy_htod(&nodes, &mut self.nodes)?;
		self.stream.memcpy_htod(&children, &mut self.children)?;
		let transitions: &[Transition] =
			bytemuck::cast_slice(transitions);
		self.stream
			.memcpy_htod(transitions, &mut self.transitions)?;

		let mut leaves_end = leaves_end as u32;
		let internals_start = leaves_end;

		if leaves_end > 10 {
			self.update_leaves(leaves_end)?;
			leaves_end = 0;
		}
		self.update_all(leaves_end, internals_start)?;

		let root = nodes.last().unwrap();
		self.update_likelihoods(
			*root,
			(root_children.0 as u32, root_children.1 as u32),
			frequencies,
		)?;

		Ok(())
	}

	/// Fetches the likelihoods calculated by [`update_likelihoods`]
	///
	/// Synchronous, blocks on the `self.likelihoods` buffer.
	///
	/// [`update_likelihoods`]: Self::update_likelihoods
	fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()> {
		self.stream.memcpy_dtoh(&self.likelihoods, patterns)?;

		let scale_sums = self.stream.clone_dtoh(&self.scale_sums)?;
		for (i, scale) in scale_sums.iter().enumerate() {
			patterns[i] -= f64::from(*scale);
		}

		Ok(())
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

		builder.arg(&self.leaves);
		builder.arg(&self.projections);
		builder.arg(&self.scales);
		builder.arg(&self.scale_sums);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.nodes);
		builder.arg(&self.children);
		builder.arg(&self.transitions);

		builder.arg(&leaves_end);
		builder.arg(&internals_start);

		// SAFETY: TODO
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

		builder.arg(&self.leaves);
		builder.arg(&self.projections);

		builder.arg(&self.nodes);
		builder.arg(&self.transitions);

		// SAFETY: TODO
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
		(left, right): (u32, u32),
		frequencies: Row,
	) -> Result<()> {
		let mut builder =
			self.stream.launch_builder(&self.update_likelihoods_fn);

		let cfg = self.cfg(32, 1);

		builder.arg(&self.projections);
		builder.arg(&self.likelihoods);
		builder.arg(&self.scales);
		builder.arg(&self.scale_sums);

		builder.arg(&root);
		builder.arg(&left);
		builder.arg(&right);
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
		let num_updated_nodes = self.num_updated_nodes + 1;

		let num_site_blocks = self.num_sites.div_ceil(128);
		let grid_dim_y = num_updated_nodes.div_ceil(128);
		let cfg = LaunchConfig {
			grid_dim: (num_site_blocks, grid_dim_y, 128),
			block_dim: (128, 1, 1),
			shared_mem_bytes: 0,
		};

		let mut builder =
			self.stream.launch_builder(&self.copy_projections_fn);

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

		builder.arg(&num_updated_nodes);
		builder.arg(&self.nodes);

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

	pub fn new(
		num_sites: usize,
		leaves: Vec<u8>,
		scale_ln: u32,
		cuda_device: usize,
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

		let scale_threshold = f64::from(-(scale_ln as i32)).exp();
		let scale_mult = f64::from(scale_ln).exp();

		let num_leaves = leaves.len() / num_sites;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let context = CudaContext::new(cuda_device)?;
		let stream = context.new_stream()?;

		// SAFETY: `CudaLikelihood` only uses a single stream, so
		// there's no need for cross-stream synchronization
		unsafe { context.disable_event_tracking() };

		let leaves = stream.clone_htod(&leaves)?;

		let projections: CudaSlice<Row> =
			stream.alloc_zeros(num_nodes * num_sites)?;
		let projections_backup: CudaSlice<Row> =
			stream.alloc_zeros(num_nodes * num_sites)?;

		let likelihoods: CudaSlice<f64> =
			stream.alloc_zeros(num_sites)?;
		let children = stream.alloc_zeros(num_internals * 2)?;
		let transitions = stream.alloc_zeros(num_edges)?;
		let nodes = stream.alloc_zeros(num_nodes)?;

		let scales = stream.alloc_zeros(num_nodes * num_sites)?;
		let scales_backup =
			stream.alloc_zeros(num_nodes * num_sites)?;
		let scale_sums = stream.alloc_zeros(num_sites)?;
		let scale_sums_backup = vec![0; num_sites];

		let opts = CompileOptions {
			include_paths: vec![
				"/usr/local/cuda/include/".to_owned()
			],
			options: vec![
				format!("-DNUM_SITES={num_sites}"),
				format!("-DNUM_LEAVES={num_leaves}"),
				format!("-DSCALE_LN={scale_ln}"),
				format!("-DSCALE_THRESHOLD={scale_threshold}"),
				format!("-DSCALE_MULT={scale_mult}"),
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
			children,
			transitions,
			nodes,

			scales,
			scales_backup,
			scale_sums,
			scale_sums_backup,

			num_sites: num_sites as u32,

			num_updated_nodes: 0,
		})
	}
}
