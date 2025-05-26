use std::fmt::{Debug, Formatter, Result};

use crate::SkVec;

impl<T> Debug for SkVec<T>
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

			if self.edited[i] {
				self.inner[i * 2].fmt(f)?;
				if self.mask[i] == 0 {
					f.write_str(" (active)")?;
				}

				f.write_str(" / ")?;

				self.inner[i * 2 + 1].fmt(f)?;
				if self.mask[i] == 1 {
					f.write_str(" (active)")?;
				}
			} else if self.mask[i] == 0 {
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
