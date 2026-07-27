//! `EpochBuf` is an epoch-versioned [`Vec`]-like structure with epoch
//! versioning.  It's designed for branchless value access and memory locality
//! between the data versions.
//!
//! The core feature, versioning, can be used via two methods.
//!
//! - [`accept`][EpochBuf::accept] confirms all of the edits done since the last
//!   epoch and drops the overwritten items.
//!
//! - [`reject`][EpochBuf::reject] rolls back all of the elements to the values
//!   they had at the start of the last epoch.
//!
//! Where an epoch is the time of creation of the vector or the last call to
//! `accept` or `reject`.
//!
//!
//! ## Example
//!
//! ```
//! use sk::EpochBuf;
//!
//! let mut v = EpochBuf::repeat(1, 3);
//! assert_eq!(v.as_ref(), [1, 1, 1]);
//!
//! v[0] = 10;
//! v[2] = 30;
//! assert_eq!(v.as_ref(), [10, 1, 30]);
//!
//! v.accept();
//! assert_eq!(v.as_ref(), [10, 1, 30]);
//!
//! v[1] = 20;
//! assert_eq!(v.as_ref(), [10, 20, 30]);
//!
//! v.reject();
//! assert_eq!(v.as_ref(), [10, 1, 30]);
//! ```

mod editbuf;
mod epoch;

pub use editbuf::EditBuf;
pub use epoch::EpochBuf;
