use anyhow::Result;

use std::io;

pub trait Write {
	fn write_array<const N: usize>(&mut self, data: [u8; N]) -> Result<()>;

	fn write_slice(&mut self, data: &[u8]) -> Result<()>;
}

// XXX: specialization + Vec<u8>

impl<T: io::Write> Write for T {
	fn write_array<const N: usize>(&mut self, data: [u8; N]) -> Result<()> {
		self.write_all(&data)?;
		Ok(())
	}

	fn write_slice(&mut self, data: &[u8]) -> Result<()> {
		self.write_all(data)?;
		Ok(())
	}
}
