# Quick GEMM benchmark review and validation limits

The error-handling changes in this PR adopt the failure cases raised in
[PR #218](https://github.com/windows-GPU-UNLOCK/Windows-GPU-UNLOCK/pull/218):
all CUDA/cuBLAS return codes are checked, only acquired resources are freed,
cleanup continues after an error, an earlier operation failure is preserved,
and any device failure makes the CLI exit nonzero.

The follow-up also addresses the input/result concerns raised in review:

- Explicit 64-bit ctypes argument and return declarations are applied before
  calling native CUDA or cuBLAS functions.
- Inputs are deterministic, non-constant FP16 matrices transferred to the GPU.
- Eight output cells are copied back and compared with a CPU float64 reference
  of the quantized inputs. A mismatch or failed copy refuses the timing result.
- CUDA events time the repeated GEMM calls. The selected device's PCI bus ID,
  workload and check limits are reported.

## GPU-free tests

```sh
python -m unittest discover -s benchmark -p 'test_quick_*.py' -v
```

These mock API successes and failures, allocation and cleanup, initialization,
event timing, output mismatches, CLI aggregation, input reproducibility, CPU
reference math, and ABI declarations. They do not validate native driver or
cuBLAS ABI behavior.

The updated runner was also executed on a Linux host with two RTX 2080 Ti GPUs
and CUDA/cuBLAS installed. `python benchmark/quick_gemm_bench.py --device -1
--runs 5` completed on both cards; all eight sampled output checks passed on
each. This is a Linux runtime smoke test only. It has no matched locked-state
baseline and does not establish unlock success or performance gain.

## Remaining limits

- Only eight output cells are checked; this is a sampled numerical check, not a
  full result validation.
- Native Windows/CUDA execution has not been validated in this repository
  change. A Windows run with the supported driver/cuBLAS versions is still
  required to validate the WinDLL path before relying on its throughput.
- This standalone timing is not a matched locked/unlocked comparison and does
  not establish unlock success. Use the repository's controlled benchmark
  protocol for performance claims.
- The GSP registry helper's behavior and causal claims were not changed or
  verified by these benchmark tests.

No firmware, registry, boot configuration or GPU state is changed by the unit
tests.
