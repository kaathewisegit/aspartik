use anyhow::Result;

pub trait Write {
	fn write_array<const N: usize>(&mut self, data: [u8; N]) -> Result<()>;

	fn write_slice(&mut self, data: &[u8]) -> Result<()>;
}

impl Write for Vec<u8> {
	fn write_array<const N: usize>(&mut self, data: [u8; N]) -> Result<()> {
		self.extend_from_slice(&data);
		Ok(())
	}

	fn write_slice(&mut self, data: &[u8]) -> Result<()> {
		self.extend_from_slice(data);
		Ok(())
	}
}
