# HyperPlonk

## Benchmark

### Proof systems

```sh
Usage: cargo bench --features benchmark --bench proof_system -- [OPTIONS]

Options:
  --system <SYSTEM>    Proof system(s) to run. [possible values: hyperplonk, halo2]
  --circuit <CIRCUIT>  Circuit to run. [possible values: standard_plonk, aggregation]
  --k <K>              (Range of) log number of rows.
```

For example to compare different proof systems on aggregation circuit, run:

```sh
cargo bench --features benchmark --bench proof_system -- \
    --system hyperplonk \
    --system halo2 \
    --circuit aggregation \
    --k 20..24
```

Then the proving time (in millisecond) will be written to `target/bench/hyperplonk` and `target/bench/halo2` respectively.
