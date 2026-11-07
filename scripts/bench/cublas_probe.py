"""Times PyTorch (cuBLAS) GEMMs on the model's matrix shapes, for a target to beat."""
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False


def bench(rows, cols, b, dtype, wdtype=None):
    w = torch.randn(rows, cols, device="cuda", dtype=wdtype or dtype)
    x = torch.randn(cols, b, device="cuda", dtype=dtype)
    times = []
    for _ in range(12):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = w @ x
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        del y
    times.sort()
    t = times[len(times) // 2]
    fma = rows * cols * b
    return t * 1e6, fma / t / 1e12


for name, rows, cols in (("lm_head", 128256, 2048), ("gate", 8192, 2048), ("down", 2048, 8192), ("q", 2048, 2048)):
    for b in (1, 8, 64, 128):
        for dt, label in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
            us, rate = bench(rows, cols, b, dt)
            print(f"{name:8s} b={b:<4d} {label}: {us:10.1f} us  {rate:7.2f} T FMA/s", flush=True)
