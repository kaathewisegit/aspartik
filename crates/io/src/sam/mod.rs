use anyhow::{bail, Error, Result};
use data::{DnaNucleotide, Phred};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamFlags(u16);

impl SamFlags {
	pub const MULTIPLE_SEGMENTS: Self = Self(0x1);
	pub const PROPERLY_ALIGNED: Self = Self(0x2);
	pub const SEGMENT_UNMAPPED: Self = Self(0x4);
	pub const NEXT_SEGMENT_UNMAPPED: Self = Self(0x8);
	pub const SEQ_REVERSE_COMPLENTED: Self = Self(0x10);
	pub const NEXT_SEQ_REVERSE_COMPLEMENTED: Self = Self(0x40);
	pub const FIRST_SEGMENT: Self = Self(0x80);
	pub const LAST_SEGMENT: Self = Self(0x100);
	pub const SECONDARY_ALIGNMENT: Self = Self(0x200);
	pub const FAILED_FILTERS: Self = Self(0x400);
	pub const SUPPLEMENTARY_ALIGNMENT: Self = Self(0x800);

	pub fn contains(&self, other: SamFlags) -> bool {
		self.0 & other.0 == other.0
	}
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarOp {
	AlignmentMatch = 0,
	Insertion = 1,
	Deletion = 2,
	SkippedRegion = 3,
	SoftClipping = 4,
	HardClipping = 5,
	Padding = 6,
	SequenceMatch = 7,
	SequenceMismatch = 8,
}

impl CigarOp {
	pub fn consumes_query(&self) -> bool {
		match self {
			CigarOp::AlignmentMatch => true,
			CigarOp::Insertion => true,
			CigarOp::Deletion => false,
			CigarOp::SkippedRegion => false,
			CigarOp::SoftClipping => true,
			CigarOp::HardClipping => false,
			CigarOp::Padding => false,
			CigarOp::SequenceMatch => true,
			CigarOp::SequenceMismatch => true,
		}
	}

	pub fn consumes_reference(&self) -> bool {
		match self {
			CigarOp::AlignmentMatch => true,
			CigarOp::Insertion => false,
			CigarOp::Deletion => true,
			CigarOp::SkippedRegion => true,
			CigarOp::SoftClipping => false,
			CigarOp::HardClipping => false,
			CigarOp::Padding => false,
			CigarOp::SequenceMatch => true,
			CigarOp::SequenceMismatch => true,
		}
	}
}

impl TryFrom<char> for CigarOp {
	type Error = Error;

	fn try_from(value: char) -> Result<Self> {
		Ok(match value {
			'M' => CigarOp::AlignmentMatch,
			'I' => CigarOp::Insertion,
			'D' => CigarOp::Deletion,
			'N' => CigarOp::SkippedRegion,
			'S' => CigarOp::SoftClipping,
			'H' => CigarOp::HardClipping,
			'P' => CigarOp::Padding,
			'=' => CigarOp::SequenceMatch,
			'X' => CigarOp::SequenceMismatch,
			_ => bail!("'{value}' is not a valid CIGAR character"),
		})
	}
}

pub struct SamSegment {
	qname: Box<str>,
	flag: SamFlags,
	/// Reference
	rname: Box<str>,
	/// 1-based leftmost mapping position
	pos: u32,
	/// Mapping quality
	mapq: u8,
	cigar: Box<[CigarOp]>,
	/// Empty if not present
	rnext: Box<str>,
	/// 0 if `rnext` is not present
	pnext: u32,
	tlen: i32,
	seq: Box<[DnaNucleotide]>,
	// XXX: convert to Phred quality
	qual: Box<[Phred]>,
}

impl SamSegment {
	pub fn qname(&self) -> &str {
		&self.qname
	}

	pub fn flag(&self) -> SamFlags {
		self.flag
	}

	pub fn rname(&self) -> &str {
		&self.rname
	}

	pub fn pos(&self) -> usize {
		self.pos as usize
	}

	pub fn mapq(&self) -> u8 {
		self.mapq
	}

	pub fn cigar(&self) -> &[CigarOp] {
		&self.cigar
	}

	pub fn rnext(&self) -> Option<&str> {
		if self.rnext.is_empty() {
			None
		} else {
			Some(&self.rnext)
		}
	}

	pub fn pnext(&self) -> Option<usize> {
		if self.rnext.is_empty() {
			None
		} else {
			Some(self.pnext as usize)
		}
	}

	pub fn tlen(&self) -> isize {
		self.tlen as isize
	}

	pub fn seq(&self) -> &[DnaNucleotide] {
		&self.seq
	}

	pub fn qual(&self) -> &[Phred] {
		&self.qual
	}

	pub fn is_mapped(&self) -> bool {
		!self.flag.contains(SamFlags::SEGMENT_UNMAPPED)
	}
}
