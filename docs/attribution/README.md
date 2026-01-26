# License attribution

## Forks

### [`statrs`](./statrs-license)

The `stats` submodule is a fork of the venerable `statrs` crate.  In
fact, most the underlying algorithms are the same.  `stats` simply
prunes most of the non-distribution functionality and adds a Python API.


### [`strsim`](./strsim-license)

Generic string similarity searches, whose code I copied and instantiated
instead of reusing the generic functions from the library.


## Dependencies

### [`anstyle`](./anstyle-license)

Interoperable ANSI style escape codes definitions, used by `clap`.


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


### [`byteorder`](./byteorder-license)

A library for endian-aware number encoding used by `rmp`.


### [`bytes`](./bytes-license)

Bytes container type I use for DNA sequences.


### [`cc`](./cc-license)

C compiler invocation utility for build scripts, used by `blake3`.


### [`clap`, `clap_builder`, `clap_lex`](./clap-license)

Command-line parser, used by `divan`.


### [`cfg-if`](./cfg-if-license)

Provides a macro which allows using `if else` with `cfg` directive.
Used by `libloading`, `parking_lot_core`, `getrandom`, and `divan`.


### [`cudarc`](./cudarc-license)

A crate which includes `sys`, unsafe idiomatic, and safe abstraction
APIs for CUDA.


### [`equivalent`](./equivalent-license)

Traits for key comparison, used by `hashbrown`.


### [`fixedbitset`](./fixedbitset-license)

Bit set collection implementation used by `petgraph`.


### [`foldhash`](./foldhash-license)

A fast non-DoS resistant hash function used by `hashbrown`.


### [`getrandom`](./getrandom-license)

A crate which fetches random data from the OS, used by `rand_core`.


### [`hashbrown`](./hashbrown-license)

The Swiss Tables implementation which powers `std`'s `HashMap`.  I use
it because it configures a faster `foldhash` and is pulled by `petgraph`
anyways.


### [`heck`](./heck-license)

Case conversion library used by `pyo3` during build time for case
conversion attributes on pyclasses.


### [`indexmap`](./indexmap-license)

Hash map which preserves insertion order, used by `petgraph`.


### [`indoc`](./indoc-license)

Macro which de-indents strings, used by `pyo3` in the `py_run` macro.


### [`inventory`](./inventory-license)

"Distributed plugin registration", which uses platform-specific
implementations to create a global list of objects, which can be
populated by different crates.  I use it via `pyo3`'s
`multiple-pymethods` feature because I create several of those with
macros.  And `#[pymethods]`, being a proc macro, doesn't support nested
macros by example.


### [`libc`](./libc-license)

Raw bindings to platform-specific C libraries.


### [`libloading`](./libloading-license)

Used by `cudarc` to dynamically load CUDA libraries.


### [`libm`](./libm-license)

Floating point emulation library, used by `num-traits` for `no_std`
targets.


### [`lock_api`](./parking_lot-license)

A library for implementing Rust-style `Mutex` using `RawMutex` types.
Used (and developed) by `parking_lot`


### [`log`](./log-license)

Unified Rust `log` facade.


### [`memchr`](./memchr-license)

SIMD-optimized string search, used by `nom` and `serde_json`.


### [`memoffset`](./memoffset-license)

Provides `offset_of`, used in `pyo3`.  It's equivalent to `std`'s
`offset_of`, but its MSRV is lower (`std`'s is 1.77).


### [`num-traits`](./num-traits-license)

Unified numerical interfaces.  Used in `linalg` and `stats`.


### [`once_cell`](./once_cell-license)

`OnceCell` implementation, also used by `pyo3`.


### [`parking_lot`](./parking_lot-license)

A great concurrency primitives library, from which I use small (1-byte)
non-poisoning `Mutex` for interior mutability in pyclasses.


### [`parking_lot_core`](./parking_lot-license)


### [`paste`](./paste-license)

**UNMAINTAINED**.  A proc macro which can create new identifiers.


### [`petgraph`](./petgraph-license)

The graph library, which will probably power the generic tree API from
`io`.


### [`ppv-lite86`](./ppv-lite86-license)

SIMD for cryptography, used by `rand_chacha`.


### [`proc-macro2`](./proc-macro2-license)

Part of Tolnay's macro suite, used by derives in `thiserror`, `serde`,
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

Randomness crates used by both `b3` and `stats`.  The PCG generator
powers the `rng` module because it's serializable.


### [`rmp`/`rmp-serde`](./rmp-license)

MessagePack implementation I use to implement pickling for Rust-based
classes.


### [`scopeguard`](./scopeguard-license)

Panic-resistant `defer`, used in `lock_api`.


### [`serde`](./serde-license)

### [`serde_derive`](./serde-license)

### [`serde_json`](./serde_json-license)

### [`smallvec`](./smallvec-license)

One-word `Vec` which stores its length and capacity in the allocation,
used by `parking_lot_core`.


### [`syn`](./syn-license)

See `proc-macro2`.


### [`target-lexicon`](./target-lexicon-license)

Library for working with host triples, used by `pyo3-build-config`.


### [`thiserror`](./thiserror-license)

`anyhow`'s sister project for deriving `Display` and `Error` on types.
Currently only used in data because `stats` still supports `no_std`.


### [`thiserror-impl`](./thiserror-license)


### [`unicode-indent`](./unicode-ident-license)

Unicode-aware identifiers check, used by `serde_derive`.


### [`unindent`](./indoc-license)

Runtime version of `indoc`.


### [`zerocopy`](./zerocopy-license)

A library for safely casting between bytes and Rust types, used by
`ppv-lite86`.


### [`divan`](./divan-license)

A convenient benchmarking library.  I picked it over `criterion` because
it allowed to set the number of iterations to one, which was useful for
long-running `b3` tests.


[`rand`]: ./rand-license
