/// Buffer with metadata for tracking edit status of items without copies
///
/// It consists of `len` bytes, for each of them:
///
/// - The first bit is a pointer to the first or the second element.
/// - The second bit is the edited state: 0 if not edited, 1 if edited.
#[derive(Debug, Clone, Default)]
pub struct EditBuf(Box<[u8]>);

impl EditBuf {
	pub fn new(len: usize) -> Self {
		Self(vec![0; len].into_boxed_slice())
	}

	#[expect(clippy::len_without_is_empty)]
	pub fn len(&self) -> usize {
		self.0.len()
	}

	/// Returns the offset, which is always 0 or 1
	///
	/// # Safety
	///
	/// `i` must be less than the length of `self`.
	pub unsafe fn offset(&self, index: usize) -> usize {
		// SAFETY: `i < self.len()` invariant
		let m = unsafe { self.0.get_unchecked(index) } & 0b1;
		usize::from(m)
	}

	/// Returns the inactive offset, which is always 0 or 1
	///
	/// # Safety
	///
	/// `i` must be less than the length of `self`.
	pub unsafe fn offset_other(&self, index: usize) -> usize {
		// SAFETY: `i < self.len()` invariant
		let offset = unsafe { self.offset(index) };
		offset ^ 0b1
	}

	pub fn accept(&mut self) {
		// zero-out the edited status
		for m in &mut self.0 {
			*m &= 0b01;
		}
	}

	pub fn reject(&mut self) {
		// zero-out the edited status and flip the offset if it was
		// present
		for m in &mut self.0 {
			// 00 -> 00
			// 01 -> 01
			// 10 -> 01
			// 11 -> 00
			*m = (*m ^ (*m >> 1)) & 1;
		}
	}

	/// `set_edited` without bounds checking
	/// # Safety
	///
	/// `index` must be less than `self.len()`.
	pub unsafe fn set_edited_unchecked(&mut self, index: usize) {
		// - If edited is 0, we set it to 1 and flip offset
		// - If edited is 1, we keep it and keep the offset
		// 00 -> 11
		// 01 -> 10
		// 10 -> 10
		// 11 -> 11
		// SAFETY: `index < self.len()` invariant
		let m = unsafe { self.0.get_unchecked_mut(index) };
		*m = ((*m & 0b01) ^ !(*m >> 1)) & 0b11;
	}

	/// Updates the pointer at `index`
	///
	/// This method is idempotent.  An already edited slot won't have its
	/// pointer changed.
	pub fn set_edited(&mut self, index: usize) {
		assert!(index < self.len());
		// SAFETY: invariant checked above
		unsafe { self.set_edited_unchecked(index) }
	}

	/// Returns `true` if at least a single element has been changed
	pub fn is_any_changed(&self) -> bool {
		self.0.iter().any(|&e| (e & 0b10) != 0)
	}

	pub fn is_changed_at(&self, index: usize) -> bool {
		(self.0[index] & 0b10) != 0
	}

	pub fn as_slice(&self) -> &[u8] {
		&self.0
	}
}
