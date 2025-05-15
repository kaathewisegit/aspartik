//! Memory leak checks

use rand::{rngs::SmallRng, Rng, SeedableRng};

use std::{cell::RefCell, mem::drop};

use skvec::{skvec, SkVec};

thread_local! {
	static COUNT: RefCell<usize> = RefCell::new(0);
}

/// `Leak` increments `COUNT` on intialization and decrements it on drop.
#[derive(Debug, PartialEq)]
struct Leak {}

impl Leak {
	fn new() -> Leak {
		COUNT.with_borrow_mut(|c| {
			println!("new {} -> {}", c, *c + 1);
			*c += 1;
		});

		Leak {}
	}
}

impl Clone for Leak {
	fn clone(&self) -> Self {
		COUNT.with_borrow_mut(|c| {
			println!("clone {} -> {}", c, *c + 1);
			*c += 1;
		});

		Leak {}
	}
}

impl Drop for Leak {
	fn drop(&mut self) {
		COUNT.with_borrow_mut(|c| {
			println!("drop {} -> {}", c, *c - 1);
			*c -= 1;
		});
	}
}

/// Runs a closure and checks that `COUNT` is zero.
fn check_leak<F>(f: F)
where
	F: FnOnce(),
{
	f();
	COUNT.with_borrow(|v| {
		assert_eq!(0, *v, "Some `Leak` structs weren't dropped")
	});
}

#[test]
fn basic() {
	check_leak(|| {
		let mut v = SkVec::new();
		v.push(Leak::new());
	});
}

#[test]
fn push() {
	check_leak(|| {
		let mut v = SkVec::new();
		v.push(Leak::new());
		v.push(Leak::new());
		v.push(Leak::new());
	});
}

#[test]
fn r#macro() {
	check_leak(|| {
		let v = skvec![Leak::new()];
		drop(v);
	});

	check_leak(|| {
		let v = skvec![Leak::new(); 10];
		drop(v);
	});
}

#[test]
fn set() {
	check_leak(|| {
		let mut v = skvec![Leak::new(); 10];
		v.set(0, Leak::new());
		v.set(9, Leak::new());
	});
}

#[test]
fn double_set() {
	check_leak(|| {
		let mut v = skvec![Leak::new()];
		v.set(0, Leak::new());
		v.set(0, Leak::new());
	});
}

#[test]
fn basic_accept() {
	check_leak(|| {
		let mut v = skvec![Leak::new(); 10];
		v.set(0, Leak::new());
		v.set(2, Leak::new());
		v.accept();
	});
}

#[test]
fn basic_reject() {
	check_leak(|| {
		let mut v = skvec![Leak::new(); 10];
		v.set(0, Leak::new());
		v.set(2, Leak::new());
		v.reject();
	});
}

#[test]
fn multi_accept_reject() {
	let mut rng = SmallRng::seed_from_u64(4);
	check_leak(move || {
		let mut v = skvec![Leak::new(); 100];
		for _ in 0..10_000 {
			// make between 0 and 50 edits
			for _ in 0..rng.random_range(0..50) {
				let index = rng.random_range(0..v.len());
				v.set(index, Leak::new());
			}

			if rng.random_bool(0.5) {
				v.accept();
			} else {
				v.reject();
			}
		}
	})
}

#[test]
fn accept_item() {
	check_leak(|| {
		let mut v = skvec![Leak::new()];
		v.set(0, Leak::new());
		v.accept_element(0);

		v.set(0, Leak::new());
		v.set(0, Leak::new());
		v.accept_element(0);
	})
}

#[test]
fn basic_clear() {
	check_leak(|| {
		let mut v = skvec![Leak::new()];
		v.clear();
	});
}

#[test]
fn edit_clear() {
	check_leak(|| {
		let mut v = skvec![Leak::new()];
		v.set(0, Leak::new());
		v.clear();
	});
}

#[test]
fn accept_clear() {
	check_leak(|| {
		let mut v = skvec![Leak::new()];
		v.set(0, Leak::new());
		v.accept();
		v.clear();
	})
}
