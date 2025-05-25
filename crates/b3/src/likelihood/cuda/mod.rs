#![allow(unused)]

use anyhow::Result;
use cudarc::{
	driver::{
		CudaContext, CudaSlice, LaunchArgs, LaunchConfig, PushKernelArg,
	},
	nvrtc::compile_ptx,
};

use super::{LikelihoodTrait, Row, Transition};

pub struct CudaLikelihood {}

const CUDA_MODULE: &str = include_str!("module.cu");

impl LikelihoodTrait<4> for CudaLikelihood {
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[usize],
		transitions: &[Transition<4>],
	) -> Result<f64> {
		todo!()
	}

	fn accept(&mut self) -> Result<()> {
		todo!()
	}

	fn reject(&mut self) -> Result<()> {
		todo!()
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

		let ptx = compile_ptx(CUDA_MODULE)?;
		let module = context.load_module(ptx)?;
		let kern = module.load_function("reject")?;

		let mut builder = stream.launch_builder(&kern);

		let num_nodes = num_nodes as u32;
		builder.arg(&num_nodes);
		builder.arg(&masks);

		let num_updated_nodes: u32 = 5;
		builder.arg(&num_updated_nodes);
		let updated_nodes: CudaSlice<u32> =
			stream.memcpy_stod(&[0, 1, 2, 3, 4])?;
		builder.arg(&updated_nodes);

		let cfg = LaunchConfig::for_num_elems(1);

		// TODO: safety
		unsafe { builder.launch(cfg) }?;

		let masks = stream.memcpy_dtov(&masks);
		println!("{masks:?}");

		todo!()
	}
}
