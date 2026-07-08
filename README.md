# pq-sqisign
[![Crate][crate-image]][crate-link]
[![Docs][docs-image]][docs-link]
![Apache2/MIT licensed][license-image]
[![Downloads][downloads-image]][crate-link]
![build](https://github.com/mikelodder7/pq-sqisign/actions/workflows/sqisign.yml/badge.svg)
![MSRV][msrv-image]

A pure-Rust implemention of SQIsign--compact post-quantum signature from quaternions and isogenies. See [spec](https://sqisign.org/spec/sqisign-20250707.pdf)

## Parameter Sets

| Parameter Set | NIST Level | Public Key | Secret Key | Signature |
| ------------- | ---------- | ---------- | ---------- | --------- |
| Level-1       | 1          | 65         | 353        | 148       |
| Level-3       | 3          | 97         | 529        | 224       |
| Level-5       | 5          | 129        | 701        | 292       |

## Performance

Signing and key generation are dominated by wide-integer multiplication in the
quaternion-lattice reduction. The internal working widths are tuned to the
minimum that preserves byte-exact KAT/C output, which is the largest available
win in pure Rust.

For maximum throughput, consumers should build with a tuned release profile —
library profiles do not propagate, so this must be set in *your* crate:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
```

and optionally `RUSTFLAGS="-C target-cpu=native"`. On Apple Silicon these add
only ~2–3% combined: the hot loop is `crypto-bigint`'s Karatsuba multiply, which
already emits the native widening-multiply (`umulh`) and whose serial carry
chains do not vectorize. A larger speedup would require x86-64 `mulx`/`adcx`/`adox`
(ADX) paths or an algorithmic change to the lattice intersection — not codegen flags.

## Warnings

#### Implementation

This implementation has not undergone any security auditing and while care has been taken no guarantees can be made for either correctness or the constant time running of the underlying functions. **Please use at your own risk.**

#### Algorithm

SQIsign is currently in the NIST PQ additional signatures 3rd round. The algorithm still requires careful security review. Please see [3.3](https://nvlpubs.nist.gov/nistpubs/ir/2026/NIST.IR.8610.pdf) for further information regarding isogeny-based signature schemes.

# License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be licensed as above, without any additional terms or
conditions.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/pq-sqisign.svg
[crate-link]: https://crates.io/crates/pq-sqisign
[docs-image]: https://docs.rs/pq-sqisign/badge.svg
[docs-link]: https://docs.rs/pq-sqisign/
[license-image]: https://img.shields.io/badge/license-Apache2.0/MIT-blue.svg
[downloads-image]: https://img.shields.io/crates/d/pq-sqisign.svg
[msrv-image]: https://img.shields.io/badge/rustc-1.95+-blue.svg
