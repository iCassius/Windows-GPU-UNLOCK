"""GPU-free tests for native-call failures, sample checks, and CLI status."""
import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import quick_gemm_bench as bench


def fake_apis():
    cuda, cublas = MagicMock(), MagicMock()
    cuda_names = (
        'cuDeviceGet', 'cuDeviceGetName', 'cuDeviceGetPCIBusId', 'cuCtxCreate_v2', 'cuMemAlloc_v2',
        'cuMemcpyHtoD_v2', 'cuMemcpyDtoH_v2', 'cuCtxSynchronize', 'cuMemFree_v2',
        'cuMemsetD8_v2', 'cuCtxDestroy_v2', 'cuEventCreate', 'cuEventRecord', 'cuEventSynchronize',
        'cuEventElapsedTime', 'cuEventDestroy_v2', 'cuInit', 'cuDeviceGetCount',
    )
    cublas_names = ('cublasCreate_v2', 'cublasSetMathMode', 'cublasGemmEx', 'cublasDestroy_v2')
    for name in cuda_names:
        getattr(cuda, name).return_value = 0
    for name in cublas_names:
        getattr(cublas, name).return_value = 0

    cuda.cuDeviceGet.side_effect = lambda out, index: (setattr(out._obj, 'value', index), 0)[1]
    cuda.cuDeviceGetName.side_effect = lambda out, size, device: (setattr(out, 'value', b'Mock GPU'), 0)[1]
    cuda.cuDeviceGetPCIBusId.side_effect = lambda out, size, device: (setattr(out, 'value', b'0000:01:00.0'), 0)[1]
    cuda.cuCtxCreate_v2.side_effect = lambda out, flags, device: (setattr(out._obj, 'value', 0x100), 0)[1]
    cublas.cublasCreate_v2.side_effect = lambda out: (setattr(out._obj, 'value', 0x200), 0)[1]

    def alloc(out, size):
        out._obj.value = 0x100000 + cuda.cuMemAlloc_v2.call_count * 0x10000
        return 0
    cuda.cuMemAlloc_v2.side_effect = alloc

    def create_event(out, flags):
        out._obj.value = 0x300 + cuda.cuEventCreate.call_count * 0x10
        return 0
    cuda.cuEventCreate.side_effect = create_event
    cuda.cuEventElapsedTime.side_effect = lambda out, start, stop: (setattr(out._obj, 'value', 1000.0), 0)[1]

    _, _, base_a, base_b = bench.build_inputs(64)
    expected = [bench.sampled_reference(base_a, base_b, 4096, row, col)
                for row, col in bench.SAMPLE_COORDINATES]
    def copy_sample(out, device_address, size):
        out._obj.value = expected[cuda.cuMemcpyDtoH_v2.call_count - 1]
        return 0
    cuda.cuMemcpyDtoH_v2.side_effect = copy_sample
    return cuda, cublas


class QuickFailureTests(unittest.TestCase):
    def run_fake(self, cuda, cublas, *, duration_ms=1000.0, **kwargs):
        if duration_ms is not None:
            cuda.cuEventElapsedTime.side_effect = lambda out, start, stop: (
                setattr(out._obj, 'value', duration_ms), 0
            )[1]
        return bench.run_gemm_benchmark(cuda, cublas, **kwargs)

    def test_each_preparation_error_refuses_result(self):
        for which, name in (('cuda', 'cuDeviceGet'), ('cuda', 'cuDeviceGetName'),
                            ('cuda', 'cuDeviceGetPCIBusId'),
                            ('cuda', 'cuCtxCreate_v2'), ('cublas', 'cublasCreate_v2'),
                            ('cublas', 'cublasSetMathMode')):
            with self.subTest(name=name):
                cuda, cublas = fake_apis()
                target = getattr(cuda if which == 'cuda' else cublas, name)
                target.side_effect = lambda *args: 7
                with self.assertRaisesRegex(RuntimeError, name):
                    self.run_fake(cuda, cublas)
                cublas.cublasGemmEx.assert_not_called()
                if name == 'cublasCreate_v2':
                    cuda.cuCtxDestroy_v2.assert_called_once()
                    cublas.cublasDestroy_v2.assert_not_called()

    def test_allocation_failure_frees_only_acquired_allocations(self):
        for prior in range(3):
            with self.subTest(prior_allocations=prior):
                cuda, cublas = fake_apis()
                def allocate(pointer, size):
                    index = cuda.cuMemAlloc_v2.call_count - 1
                    if index == prior:
                        return 2
                    pointer._obj.value = (index + 1) * 4096
                    return 0
                cuda.cuMemAlloc_v2.side_effect = allocate
                with self.assertRaisesRegex(RuntimeError, 'cuMemAlloc_v2'):
                    self.run_fake(cuda, cublas)
                self.assertEqual(cuda.cuMemFree_v2.call_count, prior)
                cublas.cublasDestroy_v2.assert_called_once()
                cuda.cuCtxDestroy_v2.assert_called_once()

    def test_copy_and_gemm_failures_refuse_result(self):
        for api, failure_index in (('cuMemcpyHtoD_v2', 0), ('cuMemcpyHtoD_v2', 1),
                                   ('cuMemsetD8_v2', 0),
                                   ('cublasGemmEx', 0), ('cublasGemmEx', 2),
                                   ('cublasGemmEx', 3), ('cublasGemmEx', 5)):
            with self.subTest(api=api, failure_index=failure_index):
                cuda, cublas = fake_apis()
                target = getattr(cublas if api.startswith('cublas') else cuda, api)
                target.side_effect = [0] * failure_index + [7]
                with self.assertRaisesRegex(RuntimeError, api):
                    self.run_fake(cuda, cublas, runs=3)
                if api == 'cublasGemmEx':
                    cuda.cuMemFree_v2.assert_called()

    def test_sync_event_and_readback_failures_refuse_result(self):
        cases = (('cuCtxSynchronize', 700), ('cuEventRecord', 700),
                 ('cuEventSynchronize', 700), ('cuEventElapsedTime', 700),
                 ('cuMemcpyDtoH_v2', 700))
        for api, status in cases:
            with self.subTest(api=api):
                cuda, cublas = fake_apis()
                getattr(cuda, api).side_effect = lambda *args: status
                with self.assertRaisesRegex(RuntimeError, api):
                    self.run_fake(cuda, cublas, duration_ms=None)
                cuda.cuMemFree_v2.assert_called()

    def test_sample_mismatch_refuses_result(self):
        cuda, cublas = fake_apis()
        cuda.cuMemcpyDtoH_v2.side_effect = lambda out, ptr, size: (
            setattr(out._obj, 'value', 999.0), 0
        )[1]
        with self.assertRaisesRegex(RuntimeError, 'sampled output mismatch'):
            self.run_fake(cuda, cublas)

    def test_cleanup_errors_do_not_hide_primary_and_all_cleanup_runs(self):
        cuda, cublas = fake_apis()
        cublas.cublasGemmEx.return_value = 7
        cuda.cuMemFree_v2.return_value = 2
        cublas.cublasDestroy_v2.return_value = 9
        with self.assertRaisesRegex(RuntimeError, '^cublasGemmEx failed'):
            self.run_fake(cuda, cublas)
        self.assertEqual(cuda.cuMemFree_v2.call_count, 3)
        cublas.cublasDestroy_v2.assert_called_once()
        cuda.cuCtxDestroy_v2.assert_called_once()

    def test_cleanup_failure_after_work_prevents_success(self):
        cuda, cublas = fake_apis()
        cuda.cuMemFree_v2.return_value = 9
        with self.assertRaisesRegex(RuntimeError, 'Cleanup failed:.*cuMemFree_v2'):
            self.run_fake(cuda, cublas)
        self.assertEqual(cuda.cuMemFree_v2.call_count, 3)
        cublas.cublasDestroy_v2.assert_called_once()
        cuda.cuCtxDestroy_v2.assert_called_once()

    def test_invalid_parameters_do_not_call_native_apis(self):
        cases = ({'device_index': -1}, {'device_index': True}, {'matrix_size': 0},
                 {'matrix_size': 65}, {'matrix_size': 23171}, {'matrix_size': True},
                 {'runs': 0}, {'runs': -1}, {'runs': True})
        for parameters in cases:
            with self.subTest(parameters=parameters):
                cuda, cublas = fake_apis()
                with self.assertRaises(ValueError):
                    self.run_fake(cuda, cublas, **parameters)
                self.assertEqual(cuda.mock_calls, [])
                self.assertEqual(cublas.mock_calls, [])

    def test_invalid_durations_and_sampled_success(self):
        for duration in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(duration=duration):
                cuda, cublas = fake_apis()
                with self.assertRaisesRegex(RuntimeError, 'duration'):
                    self.run_fake(cuda, cublas, duration_ms=duration)
        cuda, cublas = fake_apis()
        result = self.run_fake(cuda, cublas, runs=10)
        self.assertAlmostEqual(result['avg_ms'], 100.0, places=3)
        self.assertTrue(result['reference_checked'])
        self.assertEqual(result['pci_bus_id'], '0000:01:00.0')
        self.assertEqual(result['sample_count'], len(bench.SAMPLE_COORDINATES))
        self.assertEqual(cublas.cublasGemmEx.call_count, 13)
        self.assertEqual(cuda.cuMemcpyHtoD_v2.call_count, 2)
        self.assertEqual(cuda.cuMemcpyDtoH_v2.call_count, len(bench.SAMPLE_COORDINATES))


class QuickCliTests(unittest.TestCase):
    def run_cli(self, *, count=1, count_error=0, benchmark_error=None, arguments=()):
        cuda, cublas = fake_apis()
        def count_devices(pointer):
            pointer._obj.value = count
            return count_error
        cuda.cuDeviceGetCount.side_effect = count_devices
        success = {'device_index': 0, 'device_name': 'mock', 'avg_ms': 1,
                   'tflops': 1, 'sample_count': len(bench.SAMPLE_COORDINATES)}
        with patch.object(bench, 'find_cuda_libraries', return_value=(cuda, cublas)), \
                patch.object(bench, 'run_gemm_benchmark', side_effect=benchmark_error,
                             return_value=success) as run, \
                patch.object(bench.sys, 'argv', ['quick_gemm_bench.py', *arguments]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            status = bench.main()
        return status, output.getvalue(), run

    def test_failures_and_enumeration_errors_exit_nonzero(self):
        status, output, run = self.run_cli(count=2, benchmark_error=RuntimeError('simulated'))
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 2)
        self.assertNotIn('TFLOPS', output)
        status, output, run = self.run_cli(count_error=700)
        self.assertEqual(status, 1)
        self.assertIn('cuDeviceGetCount', output)
        run.assert_not_called()

    def test_later_success_does_not_hide_earlier_failure(self):
        success = {'device_index': 1, 'device_name': 'mock', 'avg_ms': 1,
                   'tflops': 1, 'sample_count': len(bench.SAMPLE_COORDINATES)}
        status, output, run = self.run_cli(count=2, benchmark_error=[RuntimeError('failed first'), success])
        self.assertEqual(status, 1)
        self.assertEqual(run.call_count, 2)
        self.assertIn('TFLOPS', output)
        self.assertIn('failed first', output)

    def test_invalid_indices_and_empty_enumeration_refuse_work(self):
        for count, arguments in ((0, ()), (1, ('--device', '1'))):
            with self.subTest(count=count, arguments=arguments):
                status, _, run = self.run_cli(count=count, arguments=arguments)
                self.assertEqual(status, 1)
                run.assert_not_called()

    def test_cli_discloses_sampled_check_limit(self):
        status, output, _ = self.run_cli()
        self.assertEqual(status, 0)
        self.assertIn('eight sampled outputs', output)
        self.assertIn('not a matched before/after unlock benchmark', output)

    def test_invalid_arguments_do_not_load_libraries(self):
        for arguments in (('--size', '0'), ('--size', '65'), ('--runs', '0'), ('--device', '-2')):
            with self.subTest(arguments=arguments), \
                    patch.object(bench, 'find_cuda_libraries') as load, \
                    patch.object(bench.sys, 'argv', ['quick_gemm_bench.py', *arguments]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    bench.main()
                self.assertEqual(result.exception.code, 2)
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
