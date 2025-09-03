# Things to implement

## `b3`

- [/] Partial likelihood scaling

  Has performance issues because of full scale copies.  Needs partial
  copies and a smarter change tracking algorithm (not a dump sum).

- [x] Dated tree tips

  - [x] Adjust all operators to respect (or check for) dated tips

- [ ] Deterministic tests with approximate tree comparisons


## `data`

- [ ] A general tree type for `io`.  And perhaps it can be used in `b3`,
  if copying is fast enough.

- [ ] Sequence alignment format (formats?)


## `io`

- [ ] Unified Python API sequence format?

- [ ] Convert parsers to Sans IO implementations

  - [x] Line-based Sans IO readers/writers which wrap Rust and Python
    readers
  - [ ] Token-based Sans IO generator?

- [ ] Non-recursive Newick parser

- [ ] Nexus parser

- [ ] VCF parser

- [ ] tskit conversion


## `linalg`

- [ ] Custom solvers?

- [ ] Heap-allocated vectors and matrices (storage types)?


## `math`

- [x] Expose Harmonic and Logistic functions to Python


## `skvec`

- [ ] Unchecked indexing
