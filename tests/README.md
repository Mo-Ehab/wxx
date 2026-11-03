# webgpu_test

Smoke/integration tests for the `wxx` WebGPU ML framework.

Run all tests:

```
python tests/run_all.py
```

Run a single file:

```
python tests/test_tensor.py
```

Each test file prints `PASS`/`FAIL` per test and exits non-zero on any failure.
Tests require a working WebGPU adapter (wgpu) and the `wxx` package on `sys.path`.
