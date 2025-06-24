#![expect(clippy::excessive_precision)]
#![forbid(unsafe_code)]

pub mod consts;
pub mod function;
pub mod prec;

// used in the `assert_almost_eq` macro
#[doc(hidden)]
pub use prec::almost_eq;
