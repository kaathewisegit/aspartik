use divan::Bencher;

use rand::{RngExt, SeedableRng, rngs::SmallRng};

use std::hint::black_box;

use skvec::SkVec;

type Row = [f64; 4];

fn data(length: usize) -> SkVec<Row> {
	let mut out = SkVec::with_capacity(length);
	let mut rng = SmallRng::seed_from_u64(4);

	for _ in 0..length {
		out.push(rng.random());
	}

	out
}

#[inline(always)]
fn edit(v: &mut SkVec<[f64; 4]>, num: usize, rng: &mut SmallRng) {
	for _ in 0..num {
		let i = rng.random_range(0..v.len());
		v.set(i, rng.random());
	}
}

#[divan::bench]
fn edit_rows_tenth_accept(bencher: Bencher) {
	bencher.with_inputs(|| black_box(data(100_000)))
		.bench_values(|mut v| {
			let mut rng = SmallRng::seed_from_u64(4);
			let num = v.len() / 10;
			edit(&mut v, num, &mut rng);
			v.accept();
		})
}

#[divan::bench]
fn edit_rows_tenth_reject(bencher: Bencher) {
	bencher.with_inputs(|| black_box(data(100_000)))
		.bench_values(|mut v| {
			let mut rng = SmallRng::seed_from_u64(4);
			let num = v.len() / 10;
			edit(&mut v, num, &mut rng);
			v.reject();
		})
}

#[divan::bench]
fn edit_rows_all_accept(bencher: Bencher) {
	bencher.with_inputs(|| black_box(data(100_000)))
		.bench_values(|mut v| {
			let mut rng = SmallRng::seed_from_u64(4);
			let num = v.len();
			edit(&mut v, num, &mut rng);
			v.accept();
		})
}

#[divan::bench]
fn edit_rows_all_reject(bencher: Bencher) {
	bencher.with_inputs(|| black_box(data(100_000)))
		.bench_values(|mut v| {
			let mut rng = SmallRng::seed_from_u64(4);
			let num = v.len();
			edit(&mut v, num, &mut rng);
			v.reject();
		})
}

fn main() {
	divan::main();
}
