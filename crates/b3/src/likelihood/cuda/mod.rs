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

	propose_fn: CudaFunction,
	reject_fn: CudaFunction,

	probabilities: CudaSlice<Row<4>>,
	masks: CudaSlice<u8>,
	likelihoods: CudaSlice<f64>,
	updated_nodes: CudaSlice<u32>,

	num_nodes: u32,
	num_sites: u32,
	num_updated_nodes: u32,
}

const CUDA_MODULE: &str = include_str!("kernels.cu");

impl LikelihoodTrait<4> for CudaLikelihood {
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[usize],
		transitions: &[Transition<4>],
	) -> Result<f64> {
		let nodes: Vec<_> = nodes.iter().map(|c| *c as u32).collect();
		let children: Vec<_> =
			children.iter().map(|c| *c as u32).collect();

		self.num_updated_nodes = nodes.len() as u32;
		self.stream.memcpy_htod(&nodes, &mut self.updated_nodes)?;
		let children = self.stream.memcpy_stod(&children)?;
		let transitions = self.stream.memcpy_stod(transitions)?;

		let mut builder = self.stream.launch_builder(&self.propose_fn);

		builder.arg(&self.num_nodes);
		builder.arg(&self.num_sites);
		builder.arg(&self.masks);
		builder.arg(&self.probabilities);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.updated_nodes);
		builder.arg(&children);
		builder.arg(&transitions);
		builder.arg(&self.likelihoods);

		let cfg = LaunchConfig::for_num_elems(self.num_sites);

		// TODO: safety
		let events = unsafe { builder.launch(cfg) }?;
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

		builder.arg(&self.num_nodes);
		builder.arg(&self.num_sites);

		builder.arg(&self.masks);

		builder.arg(&self.num_updated_nodes);
		builder.arg(&self.updated_nodes);

		let cfg = LaunchConfig::for_num_elems(self.num_sites);

		// TODO: safety
		let events = unsafe { builder.launch(cfg) }?;
		if let Some((left, right)) = events {
			self.stream.wait(&left)?;
			self.stream.wait(&right)?;
		}

		self.num_updated_nodes = 0;

		Ok(())
	}
}

impl CudaLikelihood {
	pub fn new(sites: Vec<Vec<Row<4>>>) -> Result<Self> {
		let num_sites = sites.len();
		let num_leaves = sites[0].len();
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;

		let mut probabilities = vec![];
		let mut masks: Vec<u8> = vec![];
		for column in sites {
			for row in column {
				masks.push(0);
				probabilities.push(row);
				probabilities.push(Row::default());
			}
			for _ in 0..num_internals {
				masks.push(0);
				probabilities.push(Row::default());
				probabilities.push(Row::default());
			}
		}

		let context = CudaContext::new(0)?;
		let stream = context.default_stream();

		let probabilities: CudaSlice<Row<4>> =
			stream.memcpy_stod(&probabilities)?;
		let masks: CudaSlice<u8> = stream.memcpy_stod(&masks)?;

		let likelihoods: CudaSlice<f64> =
			stream.alloc_zeros(num_sites)?;
		let updated_nodes = stream.alloc_zeros(num_nodes)?;

		let ptx = compile_ptx(CUDA_MODULE)?;
		let module = context.load_module(ptx)?;
		let reject_fn = module.load_function("reject")?;
		let propose_fn = module.load_function("propose")?;

		Ok(Self {
			stream,

			propose_fn,
			reject_fn,

			probabilities,
			masks,
			likelihoods,
			updated_nodes,

			num_updated_nodes: 0,
			num_nodes: num_nodes as u32,
			num_sites: num_sites as u32,
		})
	}
}
