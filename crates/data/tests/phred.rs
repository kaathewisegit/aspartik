use data::{Phred, seq::Character};

#[test]
fn from_ascii() {
	for ch in 0..b'!' {
		assert_eq!(Phred::from_ascii(ch), None);
	}
	for ch in b'!'..=b'I' {
		Phred::from_ascii(ch).unwrap();
	}
	for ch in b'J'..=255 {
		assert_eq!(Phred::from_ascii(ch), None);
	}
}

#[test]
fn try_from() {
	for ch in '\0'..'!' {
		Phred::try_from(ch).unwrap_err();
	}
	for ch in '!'..='I' {
		Phred::try_from(ch).unwrap();
	}
	for ch in 'J'..='ß' {
		Phred::try_from(ch).unwrap_err();
	}
}

#[test]
fn accuracy() {
	let mut last = -1.0;
	for ch in b'!'..=b'I' {
		let phred = Phred::from_ascii(ch).unwrap();
		assert!(last < phred.accuracy());
		last = phred.accuracy();
	}
}

#[test]
fn probability_incorrect() {
	let mut last = f64::INFINITY;
	for ch in b'!'..=b'I' {
		let phred = Phred::from_ascii(ch).unwrap();
		println!("{last} < {}", phred.probability_incorrect());
		assert!(last > phred.probability_incorrect());
		last = phred.probability_incorrect();
	}
}
