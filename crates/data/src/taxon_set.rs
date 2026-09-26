use picoarrow::array::{Array, ArrayUtf8, NonNullable};
#[cfg(feature = "python")]
use pyo3::{prelude::*, types::PyType};
#[cfg(feature = "python")]
use util::atomic::MonotonicUsize;

use std::{iter::FromIterator, ops::Deref, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaxonSet {
	inner: Arc<ArrayUtf8<NonNullable>>,
}

impl TaxonSet {
	pub fn new(names: ArrayUtf8<NonNullable>) -> Self {
		TaxonSet {
			inner: Arc::new(names),
		}
	}

	pub fn ranged_ints(len: usize) -> Self {
		Self::from_iter((0..len).map(|i| i.to_string()))
	}
}

impl<S> FromIterator<S> for TaxonSet
where
	S: AsRef<str>,
{
	fn from_iter<I>(names: I) -> Self
	where
		I: IntoIterator<Item = S>,
	{
		let names = names.into_iter();
		let len = names.size_hint().1.unwrap_or(0);
		let mut array = ArrayUtf8::<NonNullable>::with_capacity(len);
		for name in names {
			array.push(name.as_ref()).unwrap();
		}
		Self::new(array)
	}
}

impl From<ArrayUtf8<NonNullable>> for TaxonSet {
	fn from(value: ArrayUtf8<NonNullable>) -> Self {
		Self::new(value)
	}
}

impl Deref for TaxonSet {
	type Target = ArrayUtf8<NonNullable>;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

#[cfg(feature = "python")]
#[derive(Debug, Clone, PartialEq, Eq)]
#[pyclass(
	from_py_object,
	name = "TaxonSet",
	module = "aspartik.data",
	frozen,
	eq
)]
#[repr(transparent)]
pub struct PyTaxonSet(pub TaxonSet);

#[cfg(feature = "python")]
#[pymethods]
impl PyTaxonSet {
	#[new]
	fn new(names: Vec<String>) -> Self {
		PyTaxonSet(TaxonSet::from_iter(names))
	}

	#[classmethod]
	fn ranged_ints(_cls: Py<PyType>, len: usize) -> Self {
		PyTaxonSet(TaxonSet::ranged_ints(len))
	}

	fn __iter__(&self) -> PyTaxonSetIter {
		PyTaxonSetIter {
			taxon_set: self.0.clone(),
			index: 0.into(),
		}
	}
}

#[cfg(feature = "python")]
#[derive(Debug)]
#[pyclass(name = "TaxonSetIter", frozen)]
struct PyTaxonSetIter {
	taxon_set: TaxonSet,
	index: MonotonicUsize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyTaxonSetIter {
	fn __iter__(this: PyRef<Self>) -> PyRef<Self> {
		this
	}

	fn __next__(&self) -> Option<String> {
		let index = self.index.load();
		if index == self.taxon_set.len() {
			return None;
		}
		let out = self.taxon_set.get(index);
		self.index.add(1);
		Some(out.to_owned())
	}
}
