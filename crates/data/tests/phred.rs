use data::Phred;

#[test]
fn new() {
	for ch in 0..b'!' {
		Phred::new(ch).unwrap_err();
	}
	for ch in b'!'..=b'I' {
		Phred::new(ch).unwrap();
	}
	for ch in b'J'..=255 {
		Phred::new(ch).unwrap_err();
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
		let phred = Phred::new(ch).unwrap();
		assert!(last < phred.accuracy());
		last = phred.accuracy();
	}
}

#[test]
fn probability_incorrect() {
	let mut last = f64::INFINITY;
	for ch in b'!'..=b'I' {
		let phred = Phred::new(ch).unwrap();
		println!("{last} < {}", phred.probability_incorrect());
		assert!(last > phred.probability_incorrect());
		last = phred.probability_incorrect();
	}
}
