use std::{
	fs::File,
	io::{Read, Result as IoResult},
	path::Path,
};

pub type BoxReader = Box<dyn Read + Send + Sync>;

pub fn from_file<P: AsRef<Path>>(path: P) -> IoResult<BoxReader> {
	Ok(Box::new(File::open(path)?))
}
