use num_traits::Num;

/// Implements univariate function bisection searching for smallest `k` such
/// that `f(k) >= z`.
///
/// Returns `None` if:
///
/// - Provided interval has lower bound greater than upper bound
/// - Function was found to be non semi-monotone on the provided interval
pub fn integral_bisection_search<K: Num + Clone, T: Num + PartialOrd>(
	f: impl Fn(&K) -> T,
	z: T,
	mut lower: K,
	mut upper: K,
) -> Option<K> {
	if !(f(&lower)..=f(&upper)).contains(&z) {
		return None;
	}
	let two = K::one() + K::one();
	loop {
		let mid = (lower.clone() + upper.clone()) / two.clone();
		if !(f(&lower)..=f(&upper)).contains(&f(&mid)) {
			return None; // f found not monotone on interval
		} else if f(&lower) == z {
			return Some(lower);
		} else if f(&upper) == z || (lower.clone() + K::one()) == upper
		{
			return Some(upper); // found or no more integers between
		} else if f(&mid) >= z {
			upper = mid;
		} else {
			lower = mid;
		}
	}
}

#[cfg(test)]
mod test {
	#[test]
	fn test_integer_bisection() {
		fn search(z: usize, data: &[usize]) -> Option<usize> {
			super::integral_bisection_search(
				|idx: &usize| data[*idx],
				z,
				0,
				data.len() - 1,
			)
		}

		let needle = 3;
		let data = (0..5)
			.map(|n| if n >= needle { n + 1 } else { n })
			.collect::<Vec<_>>();

		for i in 0..(data.len()) {
			assert_eq!(search(data[i], &data), Some(i))
		}

		let infimum = search(needle, &data);
		let found_element = search(needle + 1, &data); // 4 > needle && member of range
		assert_eq!(found_element, Some(needle));
		assert_eq!(infimum, found_element)
	}
}
