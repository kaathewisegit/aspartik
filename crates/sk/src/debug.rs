use std::fmt::{Debug, Formatter, Result};

use crate::SkBuf;

impl<T> Debug for SkBuf<T>
where
	T: Debug,
{
	fn fmt(&self, f: &mut Formatter<'_>) -> Result {
		macro_rules! newline {
			() => {
				if f.alternate() {
					f.write_str("\n")?;
				}
			};
		}

		f.write_str("[")?;
		newline!();

		for i in 0..self.len() {
			if f.alternate() {
				f.write_str("    ")?;
			}

			let is_edited = self.is_changed_at(i);
			// SAFETY: i < self.len()
			let offset = unsafe { self.edits.offset_unchecked(i) };

			if is_edited && offset == 0 {
				self.items[i * 2].fmt(f)?;
				f.write_str(" (active)")?;
				f.write_str(" / ")?;
				self.items[i * 2 + 1].fmt(f)?;
			} else if is_edited && offset == 1 {
				self.items[i * 2].fmt(f)?;
				f.write_str(" / ")?;
				self.items[i * 2 + 1].fmt(f)?;
				f.write_str(" (active)")?;
			} else if !is_edited && offset == 0 {
				self.items[i * 2].fmt(f)?;
				f.write_str(" / undefined")?;
			} else if !is_edited && offset == 1 {
				f.write_str("undefined / ")?;
				self.items[i * 2 + 1].fmt(f)?;
			}

			newline!();
		}

		f.write_str("]")?;
		Ok(())
	}
}
