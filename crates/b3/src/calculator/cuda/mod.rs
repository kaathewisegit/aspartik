use anyhow::{Context, Result, anyhow, bail, ensure};
use cudarc::{
	driver::{
		CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig,
		PushKernelArg, sys::is_culib_present,
	},
	nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use parking_lot::MutexGuard;
use sk::EditBuf;

use std::{env, sync::Arc};

use super::Calculator;
use crate::{Transitions, parameters::Tree};

type Row = [f64; 4];
type Transition = [f64; 16];

const CUDA_SRC: &str =
	concat!(include_str!("typedefs.h"), include_str!("kernels.cu"),);

pub struct CudaCalculator {
	stream: Arc<CudaStream>,

	propose_fn: CudaFunction,
	update_leaves_fn: CudaFunction,
	update_likelihoods_fn: CudaFunction,

	selectors: EditBuf,
	selectors_device: CudaSlice<u8>,

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
	/// It has the length of `num_nodes * num_sites * 2`.  Each node has two
	/// `num_sites` long buffers, determined by the selector.
	projections: CudaSlice<Row>,

	/// `num_sites`-long root likelihoods
	likelihoods: CudaSlice<f64>,
	likelihoods_host: Box<[f64]>,

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

	pattern_weights: Vec<u32>,

	scales: CudaSlice<u8>,
	scale_sums: CudaSlice<u32>,
	scale_sums_backup: CudaSlice<u32>,

	/// Total number of sites
	num_patterns: u32,
}

impl Calculator<4, f64> for CudaCalculator {
	fn likelihood(
		&mut self,
		mut tree: MutexGuard<Tree>,
		transitions: &Transitions<4, f64>,
		frequencies: [f64; 4],
	) -> Result<f64> {
		let (nodes, children, leaves_end) = tree.propagation_lists();
		let tms = transitions.matrices(&nodes[..nodes.len() - 1]);

		let root_children = children.last().unwrap();
		let nodes: Vec<_> = nodes.iter().map(|&n| n as u32).collect();
		let children: Vec<_> = children
			.iter()
			.flat_map(|&[l, r]| [l as u32, r as u32])
			.collect();

		for &node in &nodes {
			self.selectors.set_edited(node as usize);
		}
		self.stream.memcpy_htod(
			self.selectors.as_slice(),
			&mut self.selectors_device,
		)?;

		self.stream.memcpy_htod(&nodes, &mut self.nodes)?;
		self.stream.memcpy_htod(&children, &mut self.children)?;
		let tms: &[Transition] = bytemuck::cast_slice(&tms);
		self.stream.memcpy_htod(tms, &mut self.transitions)?;

		let mut leaves_end = leaves_end as u32;
		let internals_start = leaves_end;

		if leaves_end > 10 {
			self.update_leaves(leaves_end)?;
			leaves_end = 0;
		}
		self.update_all(
			nodes.len() as u32 - 1,
			leaves_end,
			internals_start,
		)?;

		let root = nodes.last().unwrap();
		self.update_likelihoods(
			*root,
			(root_children[0] as u32, root_children[1] as u32),
			frequencies,
		)?;

		drop(tree);

		// blocks on the running kernels
		self.stream.memcpy_dtoh(
			&self.likelihoods,
			&mut *self.likelihoods_host,
		)?;

		let scale_sums = self.stream.clone_dtoh(&self.scale_sums)?;
		for ((likelihood, scale), weight) in self
			.likelihoods_host
			.iter_mut()
			.zip(scale_sums)
			.zip(&self.pattern_weights)
		{
			*likelihood -= f64::from(scale);
			*likelihood *= f64::from(*weight);
		}

		Ok(self.likelihoods_host.iter().sum())
	}

	fn accept(&mut self) -> Result<()> {
		self.stream.memcpy_dtod(
			&self.scale_sums,
			&mut self.scale_sums_backup,
		)?;

		self.selectors.accept();

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.stream.memcpy_dtod(
			&self.scale_sums_backup,
			&mut self.scale_sums,
		)?;

		self.selectors.reject();

		Ok(())
	}

	fn num_patterns(&self) -> usize {
		self.num_patterns as usize
	}
}

impl CudaCalculator {
	/// Updates both leaves and internal nodes
	///
	/// Asynchronous.
	fn update_all(
		&self,
		num_updated_nodes: u32,
		leaves_end: u32,
		internals_start: u32,
	) -> Result<()> {
		let mut builder = self.stream.launch_builder(&self.propose_fn);

		let block_size = 16 * 4;
		let num_pattern_blocks =
			(self.num_patterns * 4).div_ceil(block_size);
		let cfg = LaunchConfig {
			grid_dim: (num_pattern_blocks, 1, 1),
			block_dim: (block_size, 1, 1),
			shared_mem_bytes: 0,
		};

		builder.arg(&self.leaves);
		builder.arg(&self.projections);
		builder.arg(&self.scales);
		builder.arg(&self.scale_sums);
		builder.arg(&self.selectors_device);

		builder.arg(&num_updated_nodes);
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
		let num_pattern_blocks = self.num_patterns.div_ceil(block_size);
		let cfg = LaunchConfig {
			grid_dim: (num_pattern_blocks, leaves_end, 1),
			block_dim: (block_size, 4, 1),
			shared_mem_bytes: 0,
		};

		builder.arg(&self.leaves);
		builder.arg(&self.projections);
		builder.arg(&self.selectors_device);

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
		builder.arg(&self.selectors_device);

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

	fn cfg(&self, block_size: u32, dim2: u32) -> LaunchConfig {
		let num_site_blocks = self.num_patterns.div_ceil(block_size);
		LaunchConfig {
			grid_dim: (num_site_blocks, dim2, 1),
			block_dim: (block_size, 1, 1),
			shared_mem_bytes: 0,
		}
	}

	pub fn new(
		pattern_weights: Vec<u32>,
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

		let num_patterns = pattern_weights.len();
		let num_leaves = leaves.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		let context = CudaContext::new(cuda_device)?;
		// ensure there's enough space
		let required =
			num_patterns * (num_leaves + num_nodes * 8 * 4 * 2);
		let available = context.total_mem()?;
		ensure!(
			required < available,
			"The calculator requires at least {required}b of memory, but the device only has {available}b"
		);

		let stream = context.new_stream()?;

		// SAFETY: `CudaLikelihood` only uses a single stream, so
		// there's no need for cross-stream synchronization
		unsafe { context.disable_event_tracking() };

		let leaves = stream.clone_htod(&leaves)?;

		let projections: CudaSlice<Row> =
			stream.alloc_zeros(num_nodes * num_patterns * 2)?;

		let likelihoods: CudaSlice<f64> =
			stream.alloc_zeros(num_patterns)?;
		let children = stream.alloc_zeros(num_internals * 2)?;
		let transitions = stream.alloc_zeros(num_edges)?;
		let nodes = stream.alloc_zeros(num_nodes)?;

		let selectors = EditBuf::new(num_nodes);
		let selectors_device =
			stream.clone_htod(selectors.as_slice())?;

		let scales =
			stream.alloc_zeros(num_nodes * num_patterns * 2)?;
		let scale_sums = stream.alloc_zeros(num_patterns)?;
		let scale_sums_backup = stream.alloc_zeros(num_patterns)?;

		let opts = CompileOptions {
			include_paths: vec![
				format!("{}/include", cuda_path()),
				cccl_path(),
			],
			options: vec![
				format!("-DNUM_PATTERNS={num_patterns}"),
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
		let update_leaves_fn = module.load_function("update_leaves")?;
		let update_likelihoods_fn =
			module.load_function("update_likelihoods")?;

		Ok(Self {
			stream,

			propose_fn,
			update_leaves_fn,
			update_likelihoods_fn,

			selectors,
			selectors_device,

			leaves,
			projections,
			likelihoods,
			likelihoods_host: vec![0.0; num_patterns].into(),
			children,
			transitions,
			nodes,

			pattern_weights,

			scales,
			scale_sums,
			scale_sums_backup,

			num_patterns: num_patterns as u32,
		})
	}
}

fn cuda_path() -> String {
	env::var("CUDA_PATH")
		.or_else(|_| env::var("CUDA_HOME"))
		.unwrap_or_else(|_| "/usr/local/cuda".to_string())
}

fn cccl_path() -> String {
	let arch = match env::consts::ARCH {
		"x86_64" => "x86_64",
		_ => unimplemented!(),
	};
	let os = env::consts::OS;
	format!("{}/targets/{arch}-{os}/include/cccl/", cuda_path())
}
