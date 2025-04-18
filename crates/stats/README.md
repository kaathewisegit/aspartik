# statrs

Statrs provides a host of statistical utilities for Rust scientific
computing.

Included are a number of common distributions that can be sampled (i.e.
Normal, Exponential, Student's T, Gamma, Uniform, etc.) plus common
statistical functions like the gamma function, beta function, and error
function.

This library began as port of the statistical capabilities in the C#
Math.NET library.  All unit tests in the library borrowed from Math.NET
when possible and filled-in when not.  Planned for future releases are
continued implementations of distributions as well as porting over more
statistical utilities.


## Usage

Add the most recent release to your `Cargo.toml`

```toml
[dependencies]
statrs = "*" # replace * by the latest version of the crate.
```
