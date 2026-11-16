"""GPU-free tests for input construction and explicit ctypes declarations."""
import ctypes
import unittest
from unittest.mock import MagicMock

import quick_gemm_bench as bench


class QuickGemmContractTests(unittest.TestCase):
    def test_inputs_are_deterministic_finite_fp16_and_nonconstant(self):
        a0, b0, base_a0, base_b0 = bench.build_inputs(64, seed=10)
        a1, b1, _, _ = bench.build_inputs(64, seed=10)
        a2, b2, _, _ = bench.build_inputs(64, seed=11)
        self.assertEqual(a0, a1)
        self.assertEqual(b0, b1)
        self.assertNotEqual(a0, a2)
        self.assertNotEqual(b0, b2)
        self.assertEqual(len(a0), 64 * 64 * 2)
        self.assertEqual(len(b0), 64 * 64 * 2)
        self.assertGreater(len(set(a0[i:i + 2] for i in range(0, len(a0), 2))), 2)
        self.assertTrue(all(-1.0 <= value <= 1.0 for row in base_a0 for value in row))
        self.assertTrue(all(-1.0 <= value <= 1.0 for row in base_b0 for value in row))

    def test_periodic_cpu_reference_matches_direct_float64_sum(self):
        _, _, a, b = bench.build_inputs(64, seed=42)
        actual = bench.sampled_reference(a, b, 128, 7, 19)
        direct = 2 * sum(a[7][k] * b[k][19] for k in range(64))
        self.assertAlmostEqual(actual, direct, places=12)

    def test_abi_declarations_include_pointer_sized_arguments(self):
        cuda, cublas = MagicMock(), MagicMock()
        bench.configure_abis(cuda, cublas)
        self.assertIs(cuda.cuMemAlloc_v2.argtypes[1], ctypes.c_size_t)
        self.assertIs(cuda.cuMemAlloc_v2.argtypes[0]._type_, ctypes.c_uint64)
        self.assertIs(cuda.cuMemcpyHtoD_v2.argtypes[0], ctypes.c_uint64)
        self.assertIs(cuda.cuMemcpyHtoD_v2.argtypes[2], ctypes.c_size_t)
        self.assertIs(cuda.cuEventElapsedTime.restype, ctypes.c_int)
        self.assertEqual(len(cublas.cublasGemmEx.argtypes), 19)
        self.assertIs(cublas.cublasGemmEx.restype, ctypes.c_int)
        self.assertEqual(bench.CUBLAS_GEMM_DEFAULT_TENSOR_OP, 99)


if __name__ == '__main__':
    unittest.main()
