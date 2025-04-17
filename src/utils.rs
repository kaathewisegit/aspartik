#[cfg(feature = "python")]
macro_rules! impl_pyerr {
	($err: ty, $pyexc: ty) => {
		impl std::convert::From<$err> for PyErr {
			fn from(err: $err) -> PyErr {
				<$pyexc>::new_err(err)
			}
		}
	};
}

#[cfg(feature = "python")]
pub(crate) use impl_pyerr;
