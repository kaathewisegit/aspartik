# Things to implement

## `b3`

- [ ] Partial likelihood scaling

- [ ] Dated tree tips

  - [ ] Adjust all operators to respect (or check for) dated tips


## `data`

- [ ] A general tree type for `io`.  And perhaps it can be used in `b3`,
  if copying is fast enough.


## `io`

- [ ] Unified Python API sequence format?

- [ ] Convert parsers to Sans IO implementations

  - [ ] Line-based Sans IO generator which wraps Rust and Python readers
  - [ ] Token-based Sans IO generator?

- [ ] Non-recursive Newick parser


## `linalg`

- [ ] Custom solvers?

- [ ] Heap-allocated objects (storage types)?


## `math`

- [ ] Expose Harmonic and Logistic functions to Python


## `skvec`

- [ ] Unchecked indexing
