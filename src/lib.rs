#![feature(test)]
#![feature(binary_heap_into_iter_sorted)]

extern crate test;
use libc;
use std::collections::{HashMap, BinaryHeap};
use itertools::Itertools;
use std::cmp::Ordering;

/// Calculates the survival function of a conical combination of independent Bernoulli variables
/// at a given value.
///
/// # Arguments
///
/// * `probabilities` - the probabilities that the variables are non-zero.
/// * `values` - the coefficients in the combination, all of which must be positive, and
///   the length of which must equal that of `probabilities`.
/// * `k` - the value at which to evaluate the survival function.
fn internal_bernoulli_survival(probabilities: &[f64], values: &[i32], k: i32) -> f64 {
    // Calculate the sum of all the random variables by starting with the variable which
    // is 0 with probability 1, then adding each random variable one at a time.
    let mut sum = HashMap::new();
    sum.insert(0, 1.0);
    for (v, p) in values.iter().zip(probabilities) {
        // The current variable has two possible outcomes, 0 and v, which we treat
        // independently: 0 now, then v below.
        for old_p in sum.values_mut() {
            *old_p *= 1.0 - p;
        }

        for (old_v, old_p) in sum.clone().iter() {
            let new_v = old_v + v;
            // As all values are assumed to be positive, in calculating the cdf we can
            // now ignore all values that exceed the given one, as their values will
            // only ever grow when more variables are included.
            if new_v <= k {
                // The probability of seeing v is p, but we need to correct by the 1 - p we
                // included above.
                let new_p = *old_p * p / (1.0 - p);
                // Ensure that we increase the probability of the outcome new_v if it
                // already exists.
                *sum.entry(new_v).or_insert(0.0) += new_p;
            }
        }
    }

    1.0 - sum.values().sum::<f64>()
}

// test tests::bench_different_values ... bench:  11,980,840 ns/iter (+/- 864,041)
// test tests::bench_identical_values ... bench:      88,587 ns/iter (+/- 7,273)


struct ValueEntry<T> {
    prob: T,
    value: i32,
}

impl <T> Ord for ValueEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl <T> PartialOrd for ValueEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl <T> PartialEq for ValueEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl <T> Eq for ValueEntry<T> {}

// test tests::bench_different_values ... bench:  44,732,000 ns/iter (+/- 1,830,466)
// test tests::bench_identical_values ... bench:     225,832 ns/iter (+/- 10,365)
fn internal_bernoulli_survival2(probabilities: &[f64], values: &[i32], k: i32) -> f64 {
    let mut entries:Vec<_> = vec![ValueEntry::<f64>{prob: 1.0, value: 0}];

    for (this_v, this_p) in values.iter().zip(probabilities) {
        let mut new_entries:Vec<ValueEntry::<f64>> = entries
            .into_iter()
            .group_by(|v| v.value)
            .into_iter()
            .flat_map(|(old_v, entries)| {
                let old_p:f64 = entries.map(|v| v.prob).sum();
                std::iter::once(ValueEntry{prob: old_p*(1.0-this_p), value: old_v})
                    .chain(std::iter::once(ValueEntry{prob: old_p*this_p, value: this_v + old_v}))
            })
            .filter(|v| v.value<=k)
            .collect();
        new_entries.sort_unstable_by_key(|v| v.value);
        entries = new_entries;
    };
    
    let psum:f64 = entries.iter().map(|v| v.prob).sum();
    1.0 - psum
}

// test tests::bench_different_values ... bench:  37,720,180 ns/iter (+/- 1,891,642)
// test tests::bench_identical_values ... bench:     171,090 ns/iter (+/- 14,116)
fn internal_bernoulli_survival3(probabilities: &[f64], values: &[i32], k: i32) -> f64 {
    let mut entries:Vec<_> = vec![ValueEntry::<f64>{prob: 1.0, value: 0}];

    for (this_v, this_p) in values.iter().zip(probabilities) {
        let mut new_entries:Vec<ValueEntry::<f64>> = Vec::new();         
        for (old_v, entries) in entries.into_iter().group_by(|v| v.value).into_iter() {
            let old_p:f64 = entries.map(|v| v.prob).sum();
            &new_entries.push(ValueEntry{prob: old_p*(1.0-this_p), value: old_v});
            let new_v = this_v + old_v;
            if new_v <= k {
                &new_entries.push(ValueEntry{prob: old_p*this_p, value: new_v});
            }
        }
        new_entries.sort_unstable_by_key(|v| v.value);
        entries = new_entries;
    };
    
    let psum:f64 = entries.iter().map(|v| v.prob).sum();
    1.0 - psum
}

// test tests::bench_different_values ... bench:  33,801,040 ns/iter (+/- 1,845,635)
// test tests::bench_identical_values ... bench:     161,900 ns/iter (+/- 11,201)
fn internal_bernoulli_survival3B(probabilities: &[f32], values: &[i32], k: i32) -> f32 {
    let mut entries:Vec<_> = vec![ValueEntry::<f32>{prob: 1.0, value: 0}];

    for (this_v, this_p) in values.iter().zip(probabilities) {
        let mut new_entries:Vec<ValueEntry::<f32>> = Vec::new();         
        for (old_v, entries) in entries.into_iter().group_by(|v| v.value).into_iter() {
            let old_p:f32 = entries.map(|v| v.prob).sum();
            &new_entries.push(ValueEntry{prob: old_p*(1.0-this_p), value: old_v});
            let new_v = this_v + old_v;
            if new_v <= k {
                &new_entries.push(ValueEntry{prob: old_p*this_p, value: new_v});
            }
        }
        new_entries.sort_unstable_by_key(|v| v.value);
        entries = new_entries;
    };
    
    let psum:f32 = entries.iter().map(|v| v.prob).sum();
    1.0 - psum
}


// test tests::bench_different_values ... bench:  71,205,390 ns/iter (+/- 3,188,713)
// test tests::bench_identical_values ... bench:     225,810 ns/iter (+/- 11,904)
fn internal_bernoulli_survival4(probabilities: &[f64], values: &[i32], k: i32) -> f64 {
    let mut entries = BinaryHeap::new();
    entries.push(ValueEntry::<f64>{prob: 1.0, value: 0});

    for (this_v, this_p) in values.iter().zip(probabilities) {
        let mut new_entries = BinaryHeap::new();
        for (old_v, entries) in entries.into_iter_sorted().group_by(|v| v.value).into_iter() {
            let old_p:f64 = entries.map(|v| v.prob).sum();
            new_entries.push(ValueEntry{prob: old_p*(1.0-this_p), value: old_v});
            let new_v = this_v + old_v;
            if new_v <= k {
                new_entries.push(ValueEntry::<f64>{prob: old_p*this_p, value: new_v});
            }
        }
        entries = new_entries;
    };
    
    let psum:f64 = entries.iter().map(|v| v.prob).sum();
    1.0 - psum
}



#[cfg(test)]
mod tests {
    use super::internal_bernoulli_survival2 as ibs;
    
    use std::iter;
    use test::Bencher;

    #[bench]
    fn bench_identical_values(b: &mut Bencher) {
        let n = 200;
        let probabilities: Vec<_> = iter::repeat(0.5).take(n).collect();
        let values: Vec<_> = iter::repeat(10).take(n).collect();
        b.iter(|| ibs(&probabilities, &values, n as i32))
    }

    #[bench]
    fn bench_different_values(b: &mut Bencher) {
        let n = 200;
        let probabilities: Vec<_> = iter::repeat(0.5).take(n).collect();
        let values: Vec<_> = iter::repeat(1).scan(0, |acc, x| {*acc+=x; Some(*acc)}).take(n).collect();
        b.iter(|| ibs(&probabilities, &values, ((n/10)*n) as i32))
    }

    #[test]
    fn empty_variable_list_has_zero_survival_at_positive_value() {
        let probabilities = &[];
        let values = &[];
        let s = ibs(probabilities, values, 5);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn low_survival() {
        let probabilities = &[0.5, 0.5, 0.5];
        let values = &[10, 10, 10];
        let s = ibs(probabilities, values, 5);
        assert_eq!(s, 0.875);
    }

    #[test]
    fn high_survival() {
        let probabilities = &[0.5, 0.5, 0.5];
        let values = &[10, 10, 10];
        let s = ibs(probabilities, values, 25);
        assert_eq!(s, 0.125);
    }

    #[test]
    fn given_value_equals_possible_outcome_exactly() {
        let probabilities = &[0.5, 0.5, 0.5];
        let values = &[10, 10, 10];
        let s = ibs(probabilities, values, 20);
        assert_eq!(s, 0.125);
    }

    #[test]
    fn low_probability_gives_low_survival() {
        let probabilities = &[0.1, 0.2];
        let values = &[10, 20];
        let s = ibs(probabilities, values, 15);
        assert!((0.2 - s).abs() < 0.0000001);
    }

    #[test]
    fn high_probability_gives_high_survival() {
        let probabilities = &[0.8, 0.9];
        let values = &[10, 20];
        let s = ibs(probabilities, values, 15);
        assert!((0.9 - s).abs() < 0.0000001);
    }
}


/// Trivially wraps internal_bernoulli_survival to expose the function through FFI.
#[no_mangle]
pub extern "C" fn bernoulli_survival(
    size: libc::size_t,
    probability_pointer: *const f64,
    value_pointer: *const i32,
    k: i32,
) -> f64 {
    internal_bernoulli_survival(
        unsafe { std::slice::from_raw_parts(probability_pointer as *const f64, size as usize) },
        unsafe { std::slice::from_raw_parts(value_pointer as *const i32, size as usize) },
        k,
    ) as f64
}

