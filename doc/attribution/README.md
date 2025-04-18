# License attribution

## Forks

### [`statrs`](./statrs-license)

The `stats` submodule is a fork of the venerable `statrs` crate.  In
fact, most the underlying algorithms are the same.  `stats` simply
prunes most of the non-distribution functionality and adds a Python API.


## Dependencies

### [`anyhow`](./anyhow-license)

Used everywhere for error handling.  It's very convenient for Python
APIs and non-recoverable errors, because the Rust code can attach layers
of context which are more friendly than backtraces.


### [`petgraph`](./petgraph-license)

The graph library, which will probably power the generic tree API from
`io`.


### [`thiserror`](./thiserror-license)

`anyhow`'s sister project.  It's unfortunate that it doesn't support
`no_std`, as it'd be really convenient to use it for all those `stats`
`Display` implementations.


### [`nom`](./nom-license)

Parser which is currently used by `io`.  I plan to replace it at some
point in the future.


### [`bytemuck`](./bytemuck-license)

Byte casting, useful for interfacing with GPU.


### [`lapack-sys`](./lapack-sys-license)

LAPACK C interface bindings.  I couldn't get the LAPACK source crate to
work, so I might need to fork it in the future.


### [`num-traits`](./num-traits-license)

Unified numerical interfaces.  Used in `linalg` and `stats`.


### [`pyo3`](./pyo3-license)

The GOAT which provides Python inter-op.  I don't think `b3` would've
existed in its current form if it wasn't for this crate.  It'd probably
have some homegrown Luau scripting interface instead.


### [`rand`/`rand_pcg`][`rand`], [`rand_distr`]

Randomness crates, including PCG, which has a nice property of being
serializable.


### [`rayon`]

Simple parallelism library.  I used to use for multithreaded CPU
likelihood in `b3`, but the thread overhead turned to be too big.  It
seems you need at least 50k total bases to make it worth it.


### [`serde`]

### [`serde_json`]

### [`divan`]

A convenient benchmarking library.  I picked it over `criterion` because
it allowed to set the number of iterations to one, which was useful for
long-running `b3` tests.


### [`vulkano`/`vulkano-shaders`][`vulkano`]

Another gem of a library.  I was able to implement the bulk of
GPU-parallelism Felsenstein's likelihood calculator implementation in
just two evenings thanks to it.


### [`approx`]

A little floating point comparison library, useful for numerical tests.


[`rand`]: ./rand-license
[`rand_distr`]: ./rand_distr-license
[`rayon`]: ./rayon-license
[`serde`]: ./serde-license
[`serde_json`]: ./serde_json-license
[`divan`]: ./divan-license
[`vulkano`]: ./vulkano-license
[`approx`]: ./approx-license
