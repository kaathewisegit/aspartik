#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Array4 {
	inner: [f64; 4],
}

impl From<[f64; 4]> for Array4 {
	fn from(inner: [f64; 4]) -> Self {
		Array4 { inner }
	}
}

unsafe extern "C" {
	pub fn propose(
		nodes: *const u32,
		nodes_len: u32,
		children: *const u32,
		transitions: *const f64,
		leaves_end: u32,
		frequencies: Array4,
		//
		num_patterns: u32,
		samples: *const u8,
		projections: *mut f64,
		selectors: *mut u8,
		likelihoods: *mut f64,
	);
}
