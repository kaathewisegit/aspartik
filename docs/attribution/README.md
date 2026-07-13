# License attribution

## Vendoring

### [`strsim`](./strsim-license)

Generic string similarity searches, whose code I copied and instantiated
instead of reusing the generic functions from the library.


## Borrowing

### [BEAST X](./beast-license), [BEAST 2](./beast2-license)

Aside from the obvious fact that b3 is modeled after the BEAST packages,
a lot of operators and priors are 1-for-1 rewrites of their Java
counterparts.


### [BEAGLE](./beagle-license)

The current iteration of CPU likelihood calculators is basically the
BEAGLE implementation with some tweaks.


## Dependencies

### [`ahash`](./ahash-license)

Hardware-accelerated (via AES instructions) hash function.  Used by
Arrow.


### [`anyhow`](./anyhow-license)

Used everywhere for error handling.  It's very convenient for Python
APIs and non-recoverable errors, because the Rust code can attach layers
of context which are more friendly than backtraces.


### [`arbitrary`](./arbitrary-license)

Entropy-based object generation library, designed for fuzzing.  I use it
for randomized tests.


### [`arbtest`](./arbtest-license)

Simple `arbitrary`-based utility for random tests.  It has automatic
minimization, which simplifies failed tests to the smallest values
`arbitrary`-supporting types can generate which still fail.


### [`arrayref`](./arrayref-license)

Macros for taking array references to sub-slices.  Used by `blake3`.


### [`arrayvec`](./arrayvec-license)

`Vec`-like structure backed by an array without spilling.  Used by
`blake3`.


### [`arrow`](./arrow-license)

Rust implementation of Apache Arrow, including a bunch of sub-crates.


### [`autocfg`](./autocfg-license)

A build dependency used by `parking_lot`, `num-traits`, and `memoffset`,
which detects Rust compiler feature support in build scripts.


### [`bitflags`](./bitflags-license)

Macro which generates a bit flag structure with an enum-like interface.
Only used in one of `divan`'s transitive dependencies.


### [`blake3`](./blake3-license)

Implementation of a fast BLAKE3 hash function I use to uniquely identify
alignment columns.


### [`bytemuck`](./bytemuck-license)

Byte casting, which I was using for the old Vulkan calculator.  It's no
longer used anywhere, but I decided to keep it around in `linalg`.


### [`bytes`](./bytes-license)

Bytes container type I use for DNA sequences.


### [`cc`](./cc-license)

C compiler invocation utility for build scripts, used by `blake3`.


### [`cfg-if`](./cfg-if-license)

Provides a macro which allows using `if else` with `cfg` directive.
Used by `libloading`, `parking_lot_core`, `getrandom`, and `divan`.


### [`chrono`](./choro-license)

A dependency of `arrow-array` which cannot be disabled.


### [`constant_time_eq`](./constant_time_eq-license)

Used by `blake3`.


### [`cpufeatures`](./cpufeatures-license)

Used by `blake3`, a [RustCrypto](https://github.com/RustCrypto) crate.


### [`cudarc`](./cudarc-license)

A crate which includes `sys`, unsafe idiomatic, and safe abstraction
APIs for CUDA.


### [`flatbuffers`](./flatbuffers-license)

Used by Arrow as the metadata is defined as a Flatbuffers schema.


### [`fork_union`](./fork_union-license)

Another one of Mr. Vardanian's great libraries, low-latency thread
comparable with OpenMP in performance, making it suitable for
parallelising tree likelihood calculations.


### [`getrandom`](./getrandom-license)

A crate which fetches random data from the OS, used by `rand_core`.


### [`half`](./half-license)

`f16` and `bf16` types, used by Arrow.


### [`hashbrown`](./hashbrown-license)

The Swiss Tables implementation which powers `std`'s `HashMap`.  I use
it because it configures a faster `foldhash` and is pulled by `petgraph`
anyways.


### [`heck`](./heck-license)

Case conversion library used by `pyo3` during build time for case
conversion attributes on pyclasses.


### [`libc`](./libc-license)

Raw bindings to platform-specific C libraries.


### [`libloading`](./libloading-license)

Used by `cudarc` to dynamically load CUDA libraries.


### [`lock_api`](./parking_lot-license)

A library for implementing Rust-style `Mutex` using `RawMutex` types.
Used (and developed) by `parking_lot`


### [`num-traits`](./num-traits-license)

Unified numerical interfaces.  Used in `linalg`.


### [`once_cell`](./once_cell-license)

`OnceCell` implementation, also used by `pyo3`.


### [`parking_lot`](./parking_lot-license)

A great concurrency primitives library, from which I use small (1-byte)
non-poisoning `Mutex` for interior mutability in pyclasses.


### [`parking_lot_core`](./parking_lot-license)


### [`peg`](./peg-license)

Parser generator I use to parse Newick trees.


### [`pkg-config`](./pkg-config-license)

`pkg-config` API for build scripts, used by the `zstd` crate.


### [`proc-macro2`](./proc-macro2-license)

Part of Mr. Tolnay's macro suite, used by derives in `thiserror`,
`pyo3`, and `divan`.


### [`pyo3`](./pyo3-license)

A great library which provides Python inter-op.  I don't think `b3`
would've existed in its current form if it wasn't for this crate.


### [`pyo3-build-config`](./pyo3-license)

### [`pyo3-ffi`](./pyo3-license)

### [`pyo3-macros`](./pyo3-license)

### [`pyo3-macros-backend`](./pyo3-license)


### [`quote`](./quote-license)

See `proc-macro2`.


### [`rand`/`rand_pcg`][`rand`]

Randomness crates used by both `b3`.  The PCG generator powers the `rng`
module because it's serializable.


### [`rustc_version`](./rustc_version-license)

Rust compiler version for build scripts, used by Flatbuffers.


### [`scopeguard`](./scopeguard-license)

Panic-resistant `defer`, used in `lock_api`.


### [`semver`](./semver-license)

Semver parser used by `rustc_version`.


### [`smallvec`](./smallvec-license)

One-word `Vec` which stores its length and capacity in the allocation,
used by `parking_lot_core`.


### [`syn`](./syn-license)

See `proc-macro2`.


### [`target-lexicon`](./target-lexicon-license)

Library for working with host triples, used by `pyo3-build-config`.


### [`thiserror`](./thiserror-license)

`anyhow`'s sister project for deriving `Display` and `Error` on types.


### [`thiserror-impl`](./thiserror-license)


### [`unicode-indent`](./unicode-ident-license)

Unicode-aware identifiers check, used by `proc-macro2` and `syn`.


### [`zerocopy`](./zerocopy-license)

A library for safely casting between bytes and Rust types, used by
`ppv-lite86`.


### [`zstd`](./zstd-license)

Bindings to the C ZSTD implementation.


[`rand`]: ./rand-license
