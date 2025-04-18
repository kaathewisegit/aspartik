use pyo3::prelude::*;
use pyo3::{ffi::c_str, types::PyDict};

use std::ffi::CStr;

macro_rules! make_test {
	($name:ident) => {
		#[test]
		fn $name() -> PyResult<()> {
			const TEST: &CStr = c_str!(include_str!(concat!(
				"distributions/",
				stringify!($name),
				".py"
			)));

			Python::with_gil(|py| {
				let locals = PyDict::new(py);
				py.run(TEST, None, Some(&locals))
			})
		}
	};
}

make_test!(gamma);
make_test!(poisson);
