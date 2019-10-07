use crate::error::StatsError;
use crate::statistics::*;
use std::f64;

impl OrderStatistics<f64> for [f64] {
    fn order_statistic(&mut self, order: usize) -> f64 {
        let n = self.len();
        match order {
            1 => self.min(),
            _ if order == n => self.max(),
            _ if order < 1 || order > n => f64::NAN,
            _ => select_inplace(self, order - 1),
        }
    }

    fn median(&mut self) -> f64 {
        let k = self.len() / 2;
        if self.len() % 2 != 0 {
            select_inplace(self, k)
        } else {
            (select_inplace(self, k.saturating_sub(1)) + select_inplace(self, k)) / 2.0
        }
    }

    fn quantile(&mut self, tau: f64) -> f64 {
        if tau < 0.0 || tau > 1.0 || self.is_empty() {
            return f64::NAN;
        }

        let h = (self.len() as f64 + 1.0 / 3.0) * tau + 1.0 / 3.0;
        let hf = h as i64;

        if hf <= 0 || tau == 0.0 {
            return self.min();
        }
        if hf >= self.len() as i64 || tau == 1.0 {
            return self.max();
        }

        let a = select_inplace(self, (hf as usize).saturating_sub(1));
        let b = select_inplace(self, hf as usize);
        a + (h - hf as f64) * (b - a)
    }

    fn percentile(&mut self, p: usize) -> f64 {
        self.quantile(p as f64 / 100.0)
    }

    fn lower_quartile(&mut self) -> f64 {
        self.quantile(0.25)
    }

    fn upper_quartile(&mut self) -> f64 {
        self.quantile(0.75)
    }

    fn interquartile_range(&mut self) -> f64 {
        self.upper_quartile() - self.lower_quartile()
    }

    fn ranks(&mut self, tie_breaker: RankTieBreaker) -> Vec<f64> {
        let n = self.len();
        let mut ranks: Vec<f64> = vec![0.0; n];
        let mut index: Vec<usize> = (0..n).collect();

        match tie_breaker {
            RankTieBreaker::First => {
                quick_sort_all(self, &mut *index, 0, n - 1);
                unsafe {
                    for i in 0..ranks.len() {
                        ranks[*index.get_unchecked(i)] = (i + 1) as f64;
                    }
                }
                ranks
            }
            _ => {
                sort(self, &mut *index);
                let mut prev_idx = 0;
                unsafe {
                    for i in 1..n {
                        if (*self.get_unchecked(i) - *self.get_unchecked(prev_idx)).abs() <= 0.0 {
                            continue;
                        }
                        if i == prev_idx + 1 {
                            ranks[*index.get_unchecked(prev_idx)] = i as f64;
                        } else {
                            handle_rank_ties(
                                &mut *ranks,
                                &*index,
                                prev_idx as isize,
                                i as isize,
                                tie_breaker,
                            );
                        }
                        prev_idx = i;
                    }
                }

                handle_rank_ties(
                    &mut *ranks,
                    &*index,
                    prev_idx as isize,
                    n as isize,
                    tie_breaker,
                );
                ranks
            }
        }
    }
}

impl Min<f64> for [f64] {
    /// Returns the minimum value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Min;
    ///
    /// let x: [f64; 0] = [];
    /// assert!(x.min().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.min().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.min(), -2.0);
    /// ```
    fn min(&self) -> f64 {
        Statistics::min(self)
    }
}

impl Max<f64> for [f64] {
    /// Returns the maximum value in the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Max;
    ///
    /// let x: [f64; 0] = [];
    /// assert!(x.max().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.max().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.max(), 3.0);
    /// ```
    fn max(&self) -> f64 {
        Statistics::max(self)
    }
}

impl Mean<f64> for [f64] {
    /// Evaluates the sample mean, an estimate of the population
    /// mean.
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty or an entry is `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use std::f64;
    /// use statrs::statistics::Mean;
    ///
    /// # fn main() {
    /// let x = [];
    /// assert!(x.mean().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.mean().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_almost_eq!(z.mean(), 1.0 / 3.0, 1e-15);
    /// # }
    /// ```
    fn mean(&self) -> f64 {
        Statistics::mean(self)
    }
}

impl Variance<f64> for [f64] {
    /// Estimates the unbiased population variance from the provided samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's
    /// correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is
    /// `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Variance;
    ///
    /// let x = [];
    /// assert!(x.variance().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.variance().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.variance(), 19.0 / 3.0);
    /// ```
    fn variance(&self) -> f64 {
        Statistics::variance(self)
    }

    /// Estimates the unbiased population standard deviation from the provided
    /// samples
    ///
    /// # Remarks
    ///
    /// On a dataset of size `N`, `N-1` is used as a normalizer (Bessel's
    /// correction).
    ///
    /// Returns `f64::NAN` if data has less than two entries or if any entry is
    /// `f64::NAN`
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64;
    /// use statrs::statistics::Variance;
    ///
    /// let x = [];
    /// assert!(x.std_dev().is_nan());
    ///
    /// let y = [0.0, f64::NAN, 3.0, -2.0];
    /// assert!(y.std_dev().is_nan());
    ///
    /// let z = [0.0, 3.0, -2.0];
    /// assert_eq!(z.std_dev(), (19f64 / 3.0).sqrt());
    /// ```
    fn std_dev(&self) -> f64 {
        Statistics::std_dev(self)
    }
}

impl Median<f64> for [f64] {
    /// Returns the median value from the data
    ///
    /// # Remarks
    ///
    /// Returns `f64::NAN` if data is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::statistics::Median;
    ///
    /// let x = [];
    /// assert!(x.median().is_nan());
    ///
    /// let y = [0.0, 3.0, -2.0];
    /// assert_eq!(y.median(), 0.0);
    fn median(&self) -> f64 {
        let mut copy = self.to_vec();
        OrderStatistics::median(&mut *copy)
    }
}

fn handle_rank_ties(
    ranks: &mut [f64],
    index: &[usize],
    a: isize,
    b: isize,
    tie_breaker: RankTieBreaker,
) {
    let rank = match tie_breaker {
        RankTieBreaker::Average => (b + a - 1) as f64 / 2.0 + 1.0,
        RankTieBreaker::Min => (a + 1) as f64,
        RankTieBreaker::Max => b as f64,
        RankTieBreaker::First => unreachable!(),
    };
    unsafe {
        for i in a..b {
            ranks[*index.get_unchecked(i as usize)] = rank
        }
    }
}

// Selection algorithm from Numerical Recipes
// See: https://en.wikipedia.org/wiki/Selection_algorithm
fn select_inplace(arr: &mut [f64], rank: usize) -> f64 {
    if rank == 0 {
        return arr.min();
    }
    if rank > arr.len() - 1 {
        return arr.max();
    }

    unsafe {
        let mut low = 0;
        let mut high = arr.len() - 1;
        loop {
            if high <= low + 1 {
                if high == low + 1 && *arr.get_unchecked(high) < *arr.get_unchecked(low) {
                    arr.swap(low, high)
                }
                return *arr.get_unchecked(rank);
            }

            let middle = (low + high) >> 1;
            arr.swap(middle, low + 1);

            if *arr.get_unchecked(low) > *arr.get_unchecked(high) {
                arr.swap(low, high);
            }
            if *arr.get_unchecked(low + 1) > *arr.get_unchecked(high) {
                arr.swap(low + 1, high);
            }
            if *arr.get_unchecked(low) > *arr.get_unchecked(low + 1) {
                arr.swap(low, low + 1);
            }

            let mut begin = low + 1;
            let mut end = high;
            let pivot = *arr.get_unchecked(begin);
            loop {
                loop {
                    begin += 1;
                    if *arr.get_unchecked(begin) >= pivot {
                        break;
                    }
                }
                loop {
                    end -= 1;
                    if *arr.get_unchecked(end) <= pivot {
                        break;
                    }
                }
                if end < begin {
                    break;
                }
                arr.swap(begin, end);
            }

            arr[low + 1] = *arr.get_unchecked(end);
            arr[end] = pivot;

            if end >= rank {
                high = end - 1;
            }
            if end <= rank {
                low = begin;
            }
        }
    }
}

// sorts a primary slice and re-orders the secondary slice automatically. Uses
// insertion sort on small
// containers and quick sorts for larger ones
fn sort(primary: &mut [f64], secondary: &mut [usize]) {
    assert_eq!(
        primary.len(),
        secondary.len(),
        "{}",
        StatsError::ContainersMustBeSameLength
    );

    let n = primary.len();
    if n <= 1 {
        return;
    }
    if n == 2 {
        unsafe {
            if *primary.get_unchecked(0) > *primary.get_unchecked(1) {
                primary.swap(0, 1);
                secondary.swap(0, 1);
            }
            return;
        }
    }

    // insertion sort for really short containers
    if n <= 10 {
        unsafe {
            for i in 1..n {
                let key = *primary.get_unchecked(i);
                let item = *secondary.get_unchecked(i);
                let mut j = i as isize - 1;
                while j >= 0 && *primary.get_unchecked(j as usize) > key {
                    primary[j as usize + 1] = *primary.get_unchecked(j as usize);
                    secondary[j as usize + 1] = *secondary.get_unchecked(j as usize);
                    j -= 1;
                }
                primary[j as usize + 1] = key;
                secondary[j as usize + 1] = item;
            }
            return;
        }
    }

    quick_sort(primary, secondary, 0, n - 1);
}

// quick sorts a primary slice and re-orders the secondary slice automatically
fn quick_sort(primary: &mut [f64], secondary: &mut [usize], left: usize, right: usize) {
    assert_eq!(
        primary.len(),
        secondary.len(),
        "{}",
        StatsError::ContainersMustBeSameLength
    );

    // shadow left and right for mutability in loop
    let mut left = left;
    let mut right = right;

    unsafe {
        loop {
            // Pivoting
            let mut a = left;
            let mut b = right;
            let p = a + ((b - a) >> 1);

            if *primary.get_unchecked(a) > *primary.get_unchecked(p) {
                primary.swap(a, p);
                secondary.swap(a, p);
            }
            if *primary.get_unchecked(a) > *primary.get_unchecked(b) {
                primary.swap(a, b);
                secondary.swap(a, b);
            }
            if *primary.get_unchecked(p) > *primary.get_unchecked(b) {
                primary.swap(p, b);
                secondary.swap(p, b);
            }

            let pivot = *primary.get_unchecked(p);

            // Hoare partitioning
            loop {
                while *primary.get_unchecked(a) < pivot {
                    a += 1;
                }
                while pivot < *primary.get_unchecked(b) {
                    b -= 1;
                }
                if a > b {
                    break;
                }
                if a < b {
                    primary.swap(a, b);
                    secondary.swap(a, b);
                }

                a += 1;
                b -= 1;

                if a > b {
                    break;
                }
            }

            // In order to limit recursion depth to log(n), sort the shorter
            // partition recursively and the longer partition iteratively.
            //
            // Must cast to isize as it's possible for left > b or a > right/
            // TODO: make this more robust
            if (b as isize - left as isize) <= (right as isize - a as isize) {
                if left < b {
                    quick_sort(primary, secondary, left, b);
                }
                left = a;
            } else {
                if a < right {
                    quick_sort(primary, secondary, a, right);
                }
                right = b;
            }

            if left >= right {
                break;
            }
        }
    }
}

// quick sorts a primary slice and re-orders the secondary slice automatically.
// Sorts secondarily by the secondary slice on primary key duplicates
fn quick_sort_all(primary: &mut [f64], secondary: &mut [usize], left: usize, right: usize) {
    assert_eq!(
        primary.len(),
        secondary.len(),
        "{}",
        StatsError::ContainersMustBeSameLength
    );

    // shadow left and right for mutability in loop
    let mut left = left;
    let mut right = right;

    unsafe {
        loop {
            // Pivoting
            let mut a = left;
            let mut b = right;
            let p = a + ((b - a) >> 1);

            if *primary.get_unchecked(a) > *primary.get_unchecked(p)
                || *primary.get_unchecked(a) == *primary.get_unchecked(p)
                    && *secondary.get_unchecked(a) > *secondary.get_unchecked(p)
            {
                primary.swap(a, p);
                secondary.swap(a, p);
            }
            if *primary.get_unchecked(a) > *primary.get_unchecked(b)
                || *primary.get_unchecked(a) == *primary.get_unchecked(b)
                    && *secondary.get_unchecked(a) > *secondary.get_unchecked(b)
            {
                primary.swap(a, b);
                secondary.swap(a, b);
            }
            if *primary.get_unchecked(p) > *primary.get_unchecked(b)
                || *primary.get_unchecked(p) == *primary.get_unchecked(b)
                    && *secondary.get_unchecked(p) > *secondary.get_unchecked(b)
            {
                primary.swap(p, b);
                secondary.swap(p, b);
            }

            let pivot1 = *primary.get_unchecked(p);
            let pivot2 = *secondary.get_unchecked(p);

            // Hoare partitioning
            loop {
                while *primary.get_unchecked(a) < pivot1
                    || *primary.get_unchecked(a) == pivot1 && *secondary.get_unchecked(a) < pivot2
                {
                    a += 1;
                }
                while pivot1 < *primary.get_unchecked(b)
                    || pivot1 == *primary.get_unchecked(b) && pivot2 < *secondary.get_unchecked(b)
                {
                    b -= 1;
                }
                if a > b {
                    break;
                }
                if a < b {
                    primary.swap(a, b);
                    secondary.swap(a, b);
                }

                a += 1;
                b -= 1;

                if a > b {
                    break;
                }
            }

            // In order to limit recursion depth to log(n), sort the shorter
            // partition recursively and the longer partition iteratively.
            //
            // Must cast to isize as it's possible for left > b or a > right/
            // TODO: make this more robust
            if (b as isize - left as isize) <= (right as isize - a as isize) {
                if left < b {
                    quick_sort_all(primary, secondary, left, b);
                }
                left = a;
            } else {
                if a < right {
                    quick_sort_all(primary, secondary, a, right);
                }
                right = b;
            }

            if left >= right {
                break;
            }
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use crate::statistics::*;

    #[test]
    fn test_order_statistic_short() {
        let mut data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 1.0, 6.0];
        assert!(data.order_statistic(0).is_nan());
        assert_eq!(data.order_statistic(1), -3.0);
        assert_eq!(data.order_statistic(2), -1.0);
        assert_eq!(data.order_statistic(3), -0.5);
        assert_eq!(data.order_statistic(7), 5.0);
        assert_eq!(data.order_statistic(8), 6.0);
        assert_eq!(data.order_statistic(9), 10.0);
        assert!(data.order_statistic(10).is_nan());
    }

    #[test]
    fn test_quantile_short() {
        let mut data = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0, 6.0];
        assert_eq!(data.quantile(0.0), -3.0);
        assert_eq!(data.quantile(1.0), 10.0);
        assert_almost_eq!(data.quantile(0.5), 3.0 / 5.0, 1e-15);
        assert_almost_eq!(data.quantile(0.2), -4.0 / 5.0, 1e-15);
        assert_eq!(data.quantile(0.7), 137.0 / 30.0);
        assert_eq!(data.quantile(0.01), -3.0);
        assert_eq!(data.quantile(0.99), 10.0);
        assert_almost_eq!(data.quantile(0.52), 287.0 / 375.0, 1e-15);
        assert_almost_eq!(data.quantile(0.325), -37.0 / 240.0, 1e-15);
    }

    // TODO: need coverage for case where data.length > 10 to cover quick sort
    #[test]
    fn test_ranks() {
        let mut sorted_distinct = [1.0, 2.0, 4.0, 7.0, 8.0, 9.0, 10.0, 12.0];
        let mut sorted_ties = [1.0, 2.0, 2.0, 7.0, 9.0, 9.0, 10.0, 12.0];
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::Average), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::Average), [1.0, 2.5, 2.5, 4.0, 5.5, 5.5, 7.0, 8.0]);
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::Min), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::Min), [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 7.0, 8.0]);
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::Max), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::Max), [1.0, 3.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_distinct.ranks(RankTieBreaker::First), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(sorted_ties.ranks(RankTieBreaker::First), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let distinct = [1.0, 8.0, 12.0, 7.0, 2.0, 9.0, 10.0, 4.0];
        let ties = [1.0, 9.0, 12.0, 7.0, 2.0, 9.0, 10.0, 2.0];
        assert_eq!(distinct.clone().ranks(RankTieBreaker::Average), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::Average), [1.0, 5.5, 8.0, 4.0, 2.5, 5.5, 7.0, 2.5]);
        assert_eq!(distinct.clone().ranks(RankTieBreaker::Min), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::Min), [1.0, 5.0, 8.0, 4.0, 2.0, 5.0, 7.0, 2.0]);
        assert_eq!(distinct.clone().ranks(RankTieBreaker::Max), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::Max), [1.0, 6.0, 8.0, 4.0, 3.0, 6.0, 7.0, 3.0]);
        assert_eq!(distinct.clone().ranks(RankTieBreaker::First), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
        assert_eq!(ties.clone().ranks(RankTieBreaker::First), [1.0, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]);
    }

    #[test]
    fn test_median_short() {
        let even = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0, 6.0];
        assert_eq!(even.median(), 0.6);

        let odd = [-1.0, 5.0, 0.0, -3.0, 10.0, -0.5, 4.0, 0.2, 1.0];
        assert_eq!(odd.median(), 0.2);
    }

    #[test]
    fn test_median_long_constant_seq() {
        let even = vec![2.0; 100000];
        assert_eq!(2.0, even.median());

        let odd = vec![2.0; 100001];
        assert_eq!(2.0, odd.median());
    }

    // TODO: test codeplex issue 5667 (Math.NET)

    #[test]
    fn test_median_robust_on_infinities() {
        let mut data3 = [2.0, f64::NEG_INFINITY, f64::INFINITY];
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        data3 = [f64::NEG_INFINITY, 2.0, f64::INFINITY];
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        data3 = [f64::NEG_INFINITY, f64::INFINITY, 2.0];
        assert_eq!(data3.median(), 2.0);
        assert_eq!(data3.median(), 2.0);

        let data4 = [f64::NEG_INFINITY, 2.0, 3.0, f64::INFINITY];
        assert_eq!(data4.median(), 2.5);
        assert_eq!(data4.median(), 2.5);
    }
}
