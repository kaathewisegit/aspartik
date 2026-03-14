fn main() {
	println!("cargo:rerun-if-changed=src/");

	let mut build = cc::Build::new();
	build.file("src/main.c");

	if cfg!(target_os = "linux") {
		println!("cargo:rustc-link-lib=gomp");
		build.flag("-fopenmp").flag("-mavx2").flag("-mfma");
	}

	build.compile("main");
}
