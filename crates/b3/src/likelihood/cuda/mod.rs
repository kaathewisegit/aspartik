use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice};

use super::Row;

pub struct CudaLikelihood {}

impl CudaLikelihood {
	pub fn new(sites: Vec<Vec<Row<4>>>) -> Result<Self> {
		let num_sites = sites.len();

		let context = CudaContext::new(0)?;
		let stream = context.default_stream();

		let _probabilities: CudaSlice<f64> =
			stream.alloc_zeros(num_sites * 4)?;

		todo!()
	}
}
