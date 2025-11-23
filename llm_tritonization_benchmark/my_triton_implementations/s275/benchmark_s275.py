#!/usr/bin/env python3
"""
Benchmark s275: Column-wise vs Row-wise processing
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import time

from llm_triton.s275_triton_correct import s275_triton as s275_columnwise
from llm_triton.s275_triton_rowwise import s275_triton as s275_rowwise
from baselines.s275_baseline import s275_pytorch

def benchmark():
    """Benchmark both implementations"""
    test_sizes = [100, 200, 500, 1000, 2000]

    print("="*80)
    print("s275 Performance Comparison: Column-wise vs Row-wise")
    print("="*80)
    print(f"{'Size':<10} {'Column-wise (ms)':<20} {'Row-wise (ms)':<20} {'Speedup':<10}")
    print("-"*80)

    for N in test_sizes:
        # Initialize arrays
        aa = torch.randn(N, N, device='cuda', dtype=torch.float32)
        bb = torch.randn(N, N, device='cuda', dtype=torch.float32)
        cc = torch.randn(N, N, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(5):
            _ = s275_columnwise(aa.clone(), bb.clone(), cc.clone())
            _ = s275_rowwise(aa.clone(), bb.clone(), cc.clone())

        torch.cuda.synchronize()

        # Benchmark column-wise
        num_iters = 20
        start = time.perf_counter()
        for _ in range(num_iters):
            result_col = s275_columnwise(aa.clone(), bb.clone(), cc.clone())
        torch.cuda.synchronize()
        time_col = (time.perf_counter() - start) / num_iters * 1000

        # Benchmark row-wise
        start = time.perf_counter()
        for _ in range(num_iters):
            result_row = s275_rowwise(aa.clone(), bb.clone(), cc.clone())
        torch.cuda.synchronize()
        time_row = (time.perf_counter() - start) / num_iters * 1000

        # Verify correctness
        max_error = torch.max(torch.abs(result_col - result_row)).item()

        speedup = time_col / time_row
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"

        print(f"{N:<10} {time_col:<20.3f} {time_row:<20.3f} {speedup_str:<10} (err={max_error:.2e})")

    print("="*80)

if __name__ == "__main__":
    benchmark()
