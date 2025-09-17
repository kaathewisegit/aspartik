# Things to implement

## `b3`

- [x] Partial likelihood scaling

- [x] Dated tree tips

  - [x] Adjust all operators to respect (or check for) dated tips

- [ ] Deterministic tests with approximate tree comparisons

- [ ] Custom population model functions

- [ ] More birth-death models

- [ ] Python calculator objects?


## `data`

- [ ] FASTA/FASTQ

  - [ ] Python-friendly record types
  - [ ] Faster per-line streaming parsing
  - [ ] Fast parsing from whole slices?

- [ ] A general tree type for use with Newick

- [ ] Newick parser (Sans IO, non-recursive)

- [ ] Variant call format

  - [ ] Structures
  - [ ] VCF parser
  - [ ] BCF parser

- [ ] General feature format

  - [ ] Structure
  - [ ] GFF/GTF parser

- [ ] Multiple sequence alignment

  - [x] Core data type format
  - [ ] Parsing/constructors
  - [/] Views

- [ ] NEXUS

  - [ ] Data structure (arbitrary blocks?)
  - [ ] Parser


## `io`

- [ ] Python reader/writer type (from `rw`)

- [ ] A list of web APIs


## `linalg`

- [ ] Heap-allocated vectors and matrices (storage types?)

- [ ] Standalone eigenvalue algorithms


## `math`

- [ ] Tests for `float`


## `skvec`

- [ ] Unchecked indexing
