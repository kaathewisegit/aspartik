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

			if self.is_edited(i) {
				self.items[i * 2].fmt(f)?;
				if (self.metadata[i] & 0b01) == 0 {
					f.write_str(" (active)")?;
				}

				f.write_str(" / ")?;

				self.items[i * 2 + 1].fmt(f)?;
				if (self.metadata[i] & 0b01) == 1 {
					f.write_str(" (active)")?;
				}
			} else if (self.metadata[i] & 0b01) == 0 {
				f.write_str("undefined / ")?;
				self[i].fmt(f)?;
			} else {
				self[i].fmt(f)?;
				f.write_str(" / undefined")?;
			}

			if f.alternate() || i != self.len() - 1 {
				f.write_str(",")?;
				if !f.alternate() {
					f.write_str(" ")?;
				}
			}
			newline!();
		}

		f.write_str("]")?;
		Ok(())
	}
}
