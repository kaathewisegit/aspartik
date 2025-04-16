//! Provides traits for statistical computation

pub use order_statistics::*;
pub use slice_statistics::*;
pub use statistics::*;
pub use traits::*;

mod iter_statistics;
mod order_statistics;
// TODO: fix later
mod slice_statistics;
#[allow(clippy::module_inception)]
mod statistics;
mod traits;
