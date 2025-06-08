use std::{env, fs::File, path::PathBuf, process::Command};

const CUDA_PATH: &str = "src/likelihood/cuda/kernels.cu";

fn main() {
	println!("cargo::rerun-if-changed=src/likelihood/cuda/kernels.cu");

	let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
	let ptx_path = out_dir.join("kernels.ptx");

	println!(
		"cargo::rustc-env=ASPARTIK_B3_PTX_SRC_PATH={}",
		ptx_path.to_str().unwrap()
	);

	if cfg!(feature = "cuda") {
		let code = Command::new("/usr/local/cuda-12/bin/nvcc")
			.arg(CUDA_PATH)
			.arg("--ptx")
			.arg("-arch=sm_52")
			.arg("-o")
			.arg(ptx_path)
			.spawn()
			.unwrap()
			.wait()
			.unwrap();
		if !code.success() {
			println!("cargo::error={code:?}");
		}
	} else {
		// create a dummy file so that compilation doesn't fail
		File::create(ptx_path).unwrap();
	}
}
