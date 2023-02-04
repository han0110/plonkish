# HyperPlonk

## Benchmark

### Proof systems

```sh
Usage: cargo bench --bench proof_system -- [OPTIONS]

Options:
  --system <SYSTEM>    Proof system(s) to run. [possible values: hyperplonk, halo2, espresso_hyperplonk]
  --circuit <CIRCUIT>  Circuit to run. [possible values: standard_plonk, aggregation]
  --k <K>              (Range of) log number of rows.
```

For example to compare different proof systems on standard PLONK, run:

```sh
cargo bench --bench proof_system -- \
    --system hyperplonk \
    --system halo2 \
    --system espresso_hyperplonk \
    --circuit standard_plonk \
    --k 20..24
```

Then the proving time (in millisecond) will be written to `target/bench/{hyperplonk,halo2,espresso_hyperplonk}` respectively.

To further see cost breakdown of proving time, run the same bench commanad with an extra cargo flag `--features timer`, then pipe the output to plotter `cargo run plotter -- -`, and the result will be rendered in `target/bench`. For example:

```sh
cargo bench --bench proof_system --features timer -- ... \
  | cargo run plotter -- -
```
