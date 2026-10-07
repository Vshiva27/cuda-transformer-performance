# benchmarks/

Results of real benchmark runs, one folder per GPU, created by `scripts/run_all.sh`:

```
benchmarks/<GPU_name>/
├── environment.txt      GPU, driver, CUDA, compiler, PyTorch versions; clocks at start
├── tests.txt            correctness tests (the run stops if any fails)
├── vector_add.txt/.csv  one pair per C++ benchmark: printed tables + machine-readable rows
├── matmul.txt/.csv
├── softmax.txt/.csv
├── layernorm.txt/.csv
├── precision.txt/.csv
├── attention.txt/.csv
├── pytorch.txt/.csv     PyTorch baseline (python/benchmark.py)
└── summary.md           all tables + headline numbers (python/summarize.py)
```

Nothing in this folder is written by hand. `docs/12_results.md` quotes numbers from these files.
