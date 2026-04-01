use anyhow::{Result, bail};

use core::mem;

pub trait Read<'a> {
	fn read_array<const N: usize>(&mut self) -> Result<[u8; N]>;

	fn read_slice(&mut self, len: usize) -> Result<&'a [u8]>;
}

impl<'a> Read<'a> for &'a [u8] {
	fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
		let Some((chunk, rest)) =
			mem::take(self).split_first_chunk::<N>()
		else {
			bail!(
				"Tried to read {} bytes from a slice the size of {}",
				N,
				self.len()
			);
		};
		*self = rest;
		Ok(*chunk)
	}

	fn read_slice(&mut self, len: usize) -> Result<&'a [u8]> {
		let Some((read, rest)) = mem::take(self).split_at_checked(len)
		else {
			bail!(
				"Tried to read {} bytes from a slice the size of {}",
				len,
				self.len()
			);
		};
		*self = rest;
		Ok(read)
	}
}
