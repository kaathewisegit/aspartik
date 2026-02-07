use std::{
	fs::File,
	io::{Read, Result as IoResult, Write},
	path::Path,
};

pub enum AnyReader {
	File(File),
	Dynamic(Box<dyn Read + Send + Sync>),
}

impl AnyReader {
	pub fn from_file<P: AsRef<Path>>(path: P) -> IoResult<Self> {
		File::open(path.as_ref()).map(AnyReader::File)
	}

	pub fn from_dynamic<R>(reader: R) -> Self
	where
		R: Read + Send + Sync + 'static,
	{
		AnyReader::Dynamic(Box::new(reader))
	}
}

impl Read for AnyReader {
	fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
		match self {
			AnyReader::File(f) => f.read(buf),
			AnyReader::Dynamic(reader) => reader.read(buf),
		}
	}
}

pub enum AnyWriter {
	File(File),
	Rust(Box<dyn Write + Send + Sync>),
}

impl AnyWriter {
	pub fn from_dynamic<W>(writer: W) -> Self
	where
		W: Write + Send + Sync + 'static,
	{
		AnyWriter::Rust(Box::new(writer))
	}
}

impl Write for AnyWriter {
	fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
		match self {
			AnyWriter::File(f) => f.write(buf),
			AnyWriter::Rust(writer) => writer.write(buf),
		}
	}

	fn flush(&mut self) -> IoResult<()> {
		match self {
			AnyWriter::File(f) => f.flush(),
			AnyWriter::Rust(writer) => writer.flush(),
		}
	}
}
