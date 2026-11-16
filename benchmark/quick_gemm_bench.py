"""Compiler-free FP16 GEMM timing with sampled numerical verification.

This experimental diagnostic calls the installed CUDA Driver and cuBLAS via
ctypes. It initializes deterministic inputs and checks sampled FP32 outputs
against a CPU float64 reference. It is not a substitute for a matched unlock
benchmark or a full numerical correctness test.
"""

import argparse
import ctypes
import math
import os
from pathlib import Path
import random
import struct
import sys


CUDA_R_32F = 0
CUDA_R_16F = 2
CUBLAS_TENSOR_OP_MATH = 1
CUBLAS_COMPUTE_32F = 68
CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
INPUT_PERIOD = 64
SAMPLE_COORDINATES = ((0, 0), (1, 3), (7, 29), (17, 55),
                      (31, 4), (43, 21), (52, 12), (63, 63))
_DLL_DIRECTORY_HANDLES = []


def find_cuda_libraries():
    """Locate the CUDA driver and cuBLAS libraries on Windows or Linux."""
    script_dir = Path(__file__).resolve().parent
    cuda_lib = None
    cublas_lib = None

    if sys.platform == "win32":
        search_dirs = [
            script_dir,
            script_dir.parent / "tools",
            Path(os.environ.get("CUDA_PATH", "")) / "bin",
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
        ]
        for directory in search_dirs:
            if directory.exists():
                try:
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
                except (AttributeError, OSError):
                    pass
        try:
            cuda_lib = ctypes.WinDLL("nvcuda.dll")
        except OSError:
            pass
        candidates = (
            "cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll",
            str(script_dir / "cublas64_12.dll"),
            str(script_dir / "cublas64_11.dll"),
            str(script_dir.parent / "tools" / "cublas64_12.dll"),
            str(script_dir.parent / "tools" / "cublas64_11.dll"),
        )
        for candidate in candidates:
            try:
                cublas_lib = ctypes.WinDLL(candidate)
                break
            except OSError:
                continue
    else:
        for candidate in ("libcuda.so.1", "libcuda.so"):
            try:
                cuda_lib = ctypes.CDLL(candidate)
                break
            except OSError:
                pass
        for candidate in ("libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"):
            try:
                cublas_lib = ctypes.CDLL(candidate)
                break
            except OSError:
                pass
    return cuda_lib, cublas_lib


def _set_signature(library, name, argtypes):
    function = getattr(library, name)
    function.argtypes = argtypes
    function.restype = ctypes.c_int


def configure_abis(cuda, cublas):
    """Set explicit 64-bit ctypes signatures for every called native function."""
    i, u, size = ctypes.c_int, ctypes.c_uint, ctypes.c_size_t
    p_i = ctypes.POINTER(i)
    p_char = ctypes.POINTER(ctypes.c_char)
    p_void = ctypes.POINTER(ctypes.c_void_p)
    p_u64 = ctypes.POINTER(ctypes.c_uint64)
    p_float = ctypes.POINTER(ctypes.c_float)
    vptr, u64 = ctypes.c_void_p, ctypes.c_uint64

    cuda_functions = (
        ("cuInit", [u]),
        ("cuDeviceGetCount", [p_i]),
        ("cuDeviceGet", [p_i, i]),
        ("cuDeviceGetName", [p_char, i, i]),
        ("cuDeviceGetPCIBusId", [p_char, i, i]),
        ("cuCtxCreate_v2", [p_void, u, i]),
        ("cuCtxDestroy_v2", [vptr]),
        ("cuCtxSynchronize", []),
        ("cuMemAlloc_v2", [p_u64, size]),
        ("cuMemFree_v2", [u64]),
        ("cuMemsetD8_v2", [u64, ctypes.c_ubyte, size]),
        ("cuMemcpyHtoD_v2", [u64, vptr, size]),
        ("cuMemcpyDtoH_v2", [vptr, u64, size]),
        ("cuEventCreate", [p_void, u]),
        ("cuEventRecord", [vptr, vptr]),
        ("cuEventSynchronize", [vptr]),
        ("cuEventElapsedTime", [p_float, vptr, vptr]),
        ("cuEventDestroy_v2", [vptr]),
    )
    for name, argtypes in cuda_functions:
        _set_signature(cuda, name, argtypes)

    cublas_functions = (
        ("cublasCreate_v2", [p_void]),
        ("cublasDestroy_v2", [vptr]),
        ("cublasSetMathMode", [vptr, i]),
        ("cublasGemmEx", [vptr, i, i, i, i, i, vptr, vptr, i, i,
                           vptr, i, i, vptr, vptr, i, i, i, i]),
    )
    for name, argtypes in cublas_functions:
        _set_signature(cublas, name, argtypes)


def checked(status, operation):
    """Never turn a failed CUDA/cuBLAS call into a timing result."""
    if status != 0:
        raise RuntimeError(f"{operation} failed (status {status})")


def validate_parameters(device_index, matrix_size, runs):
    if type(device_index) is not int or device_index < 0:
        raise ValueError("Device index must be a non-negative integer")
    maximum = math.isqrt(((1 << 31) - 1) // 4)
    if (type(matrix_size) is not int or matrix_size < INPUT_PERIOD or
            matrix_size > maximum or matrix_size % INPUT_PERIOD):
        raise ValueError(
            f"Matrix dimension must be a multiple of {INPUT_PERIOD} "
            f"from {INPUT_PERIOD} to {maximum}"
        )
    if type(runs) is not int or runs < 1:
        raise ValueError("Benchmark iterations must be a positive integer")


def build_inputs(matrix_size, seed=12345):
    """Create finite, deterministic, non-constant FP16 matrices in column-major order."""
    validate_parameters(0, matrix_size, 1)
    rng = random.Random(seed)
    base_a = [[struct.unpack("<e", struct.pack("<e", rng.uniform(-1.0, 1.0)))[0]
               for _ in range(INPUT_PERIOD)] for _ in range(INPUT_PERIOD)]
    base_b = [[struct.unpack("<e", struct.pack("<e", rng.uniform(-1.0, 1.0)))[0]
               for _ in range(INPUT_PERIOD)] for _ in range(INPUT_PERIOD)]

    def make_matrix(base):
        period_columns = []
        for col in range(INPUT_PERIOD):
            column = b"".join(struct.pack("<e", base[row][col])
                              for row in range(INPUT_PERIOD))
            period_columns.append(column * (matrix_size // INPUT_PERIOD))
        return b"".join(period_columns) * (matrix_size // INPUT_PERIOD)

    return make_matrix(base_a), make_matrix(base_b), base_a, base_b


def sampled_reference(base_a, base_b, matrix_size, row, col):
    """Compute one output in float64 from the quantized periodic FP16 inputs."""
    period = len(base_a)
    return (sum(base_a[row % period][k] * base_b[k][col % period]
                for k in range(period)) * (matrix_size // period))


def _resource_cleanup(resources, primary_error):
    """Try every acquired resource cleanup and fail if any cleanup fails."""
    errors = []
    for function, argument, operation in reversed(resources):
        try:
            checked(function(argument), operation)
        except Exception as error:
            errors.append(str(error))
    if errors and primary_error is None:
        raise RuntimeError("Cleanup failed: " + "; ".join(errors))


def run_gemm_benchmark(cuda, cublas, device_index=0, matrix_size=4096, runs=10):
    """Run one FP16-input/FP32-accumulate GEMM with sampled output validation."""
    validate_parameters(device_index, matrix_size, runs)
    configure_abis(cuda, cublas)

    dev = ctypes.c_int()
    checked(cuda.cuDeviceGet(ctypes.byref(dev), device_index), "cuDeviceGet")
    name_buf = ctypes.create_string_buffer(256)
    checked(cuda.cuDeviceGetName(name_buf, len(name_buf), dev), "cuDeviceGetName")
    device_name = name_buf.value.decode("utf-8", errors="replace")
    pci_buf = ctypes.create_string_buffer(32)
    checked(cuda.cuDeviceGetPCIBusId(pci_buf, len(pci_buf), dev), "cuDeviceGetPCIBusId")
    pci_bus_id = pci_buf.value.decode("ascii", errors="replace")

    resources = []
    primary_error = None
    try:
        context = ctypes.c_void_p()
        checked(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, dev), "cuCtxCreate_v2")
        resources.append((cuda.cuCtxDestroy_v2, context, "cuCtxDestroy_v2"))

        handle = ctypes.c_void_p()
        checked(cublas.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate_v2")
        resources.append((cublas.cublasDestroy_v2, handle, "cublasDestroy_v2"))
        checked(cublas.cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode")

        n = matrix_size
        d_a, d_b, d_c = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        for pointer, allocation_size in ((d_a, n * n * 2), (d_b, n * n * 2), (d_c, n * n * 4)):
            checked(cuda.cuMemAlloc_v2(ctypes.byref(pointer), allocation_size), "cuMemAlloc_v2")
            resources.append((cuda.cuMemFree_v2, pointer, "cuMemFree_v2"))

        a_bytes, b_bytes, base_a, base_b = build_inputs(n)
        a_host = ctypes.create_string_buffer(a_bytes)
        b_host = ctypes.create_string_buffer(b_bytes)
        checked(cuda.cuMemcpyHtoD_v2(d_a, a_host, len(a_bytes)), "cuMemcpyHtoD_v2(A)")
        checked(cuda.cuMemcpyHtoD_v2(d_b, b_host, len(b_bytes)), "cuMemcpyHtoD_v2(B)")
        checked(cuda.cuMemsetD8_v2(d_c, 0xff, n * n * 4), "cuMemsetD8_v2(C sentinel)")
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)

        def gemm():
            checked(cublas.cublasGemmEx(
                handle, 0, 0, n, n, n,
                ctypes.byref(alpha), ctypes.c_void_p(d_a.value), CUDA_R_16F, n,
                ctypes.c_void_p(d_b.value), CUDA_R_16F, n,
                ctypes.byref(beta), ctypes.c_void_p(d_c.value), CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            ), "cublasGemmEx")

        for _ in range(3):
            gemm()
        checked(cuda.cuCtxSynchronize(), "cuCtxSynchronize (warmup)")

        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        checked(cuda.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)")
        resources.append((cuda.cuEventDestroy_v2, start, "cuEventDestroy_v2(start)"))
        checked(cuda.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate(stop)")
        resources.append((cuda.cuEventDestroy_v2, stop, "cuEventDestroy_v2(stop)"))
        checked(cuda.cuEventRecord(start, None), "cuEventRecord(start)")
        for _ in range(runs):
            gemm()
        checked(cuda.cuEventRecord(stop, None), "cuEventRecord(stop)")
        checked(cuda.cuEventSynchronize(stop), "cuEventSynchronize(stop)")
        elapsed_ms = ctypes.c_float()
        checked(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start, stop), "cuEventElapsedTime")
        if not math.isfinite(elapsed_ms.value) or elapsed_ms.value <= 0:
            raise RuntimeError("Non-positive or non-finite benchmark duration")

        samples = []
        for row, col in SAMPLE_COORDINATES:
            value = ctypes.c_float()
            offset = (row + col * n) * ctypes.sizeof(ctypes.c_float)
            checked(cuda.cuMemcpyDtoH_v2(ctypes.byref(value), d_c.value + offset,
                                         ctypes.sizeof(value)), "cuMemcpyDtoH_v2(sample)")
            expected = sampled_reference(base_a, base_b, n, row, col)
            tolerance = 2e-3 + 2e-3 * abs(expected)
            if not math.isfinite(value.value) or abs(value.value - expected) > tolerance:
                raise RuntimeError(
                    f"sampled output mismatch at ({row},{col}): "
                    f"got {value.value}, expected {expected:.7g}, tolerance {tolerance:.3g}"
                )
            samples.append({"row": row, "column": col, "actual": value.value,
                            "expected": expected, "tolerance": tolerance})

        avg_ms = elapsed_ms.value / runs
        tflops = (2 * n ** 3 / (avg_ms / 1000.0)) / 1e12
        if not math.isfinite(tflops) or tflops <= 0:
            raise RuntimeError("Non-positive or non-finite benchmark throughput")
        return_value = {
            "device_index": device_index,
            "device_name": device_name,
            "pci_bus_id": pci_bus_id,
            "matrix_size": matrix_size,
            "avg_ms": avg_ms,
            "tflops": tflops,
            "input_dtype": "FP16",
            "accumulation_dtype": "FP32",
            "timing_method": "CUDA events over repeated cuBLAS GEMMs",
            "reference_checked": True,
            "reference_check": "sampled FP32 outputs vs CPU float64 reference",
            "sample_count": len(samples),
            "samples": samples,
        }
    except Exception as error:
        primary_error = error
        raise
    finally:
        _resource_cleanup(resources, primary_error)
    return return_value


def main():
    parser = argparse.ArgumentParser(description="Experimental CUDA/cuBLAS GEMM diagnostic")
    parser.add_argument("--device", type=int, default=-1, help="CUDA device index (-1 for all)")
    parser.add_argument("--size", type=int, default=4096, help="Matrix dimension N (multiple of 64)")
    parser.add_argument("--runs", type=int, default=10, help="Timed GEMM iterations")
    args = parser.parse_args()
    try:
        if args.device < -1:
            raise ValueError("Device must be -1 (all) or a non-negative index")
        validate_parameters(0 if args.device == -1 else args.device, args.size, args.runs)
    except ValueError as error:
        parser.error(str(error))

    cuda, cublas = find_cuda_libraries()
    if not cuda or not cublas:
        print("[ERROR] Could not load CUDA driver or cuBLAS library.")
        print("        An NVIDIA driver and a compatible cuBLAS library are required.")
        return 1
    try:
        configure_abis(cuda, cublas)
        checked(cuda.cuInit(0), "cuInit")
        count = ctypes.c_int()
        checked(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    except (AttributeError, OSError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        return 1
    if count.value < 1:
        print("[ERROR] No NVIDIA devices found.")
        return 1
    if args.device >= count.value:
        print("[ERROR] Requested device index is outside the enumerated GPU set.")
        return 1

    devices_to_test = range(count.value) if args.device < 0 else [args.device]
    print(f"Running {args.size}x{args.size} FP16 GEMM with FP32 accumulation on {len(devices_to_test)} GPU(s)...")
    print("[CHECK] Inputs are initialized; eight sampled outputs are checked against a CPU float64 reference.")
    print("[LIMIT] This diagnostic is not a matched before/after unlock benchmark or full numerical certification.")
    failed = False
    for index in devices_to_test:
        try:
            result = run_gemm_benchmark(cuda, cublas, index, args.size, args.runs)
            print(f"GPU {result['device_index']} [{result['device_name']}]: "
                  f"{result['avg_ms']:.3f} ms | {result['tflops']:.2f} TFLOPS "
                  f"| sampled reference PASS ({result['sample_count']} outputs)")
        except Exception as error:
            failed = True
            print(f"GPU {index} benchmark failed: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
