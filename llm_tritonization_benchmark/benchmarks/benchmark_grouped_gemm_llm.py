"""
Grouped GEMM Benchmark: Baseline vs Expert vs LLM Triton

Three-way comparison:
1. PyTorch baseline (loop with separate launches)
2. Expert Triton (from official tutorial)
3. LLM-generated Triton (with hardware context)
"""

import torch
import time
import sys
import os

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, "../baselines")
sys.path.insert(0, "../llm_triton")

from grouped_gemm_baseline import grouped_gemm_baseline
from grouped_gemm_triton_llm import grouped_gemm_llm

# Import expert implementation
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel_expert(
    group_a_ptrs, group_b_ptrs, group_c_ptrs,
    group_gemm_sizes, g_lds, group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            tl.store(c_ptrs, c)

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


def grouped_gemm_expert(group_A, group_B):
    """Expert Triton from tutorial"""
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs, B_addrs, C_addrs, g_sizes, g_lds, group_C = [], [], [], [], [], []
    for A, B in zip(group_A, group_B):
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device='cuda', dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device='cuda')
    d_b_ptrs = torch.tensor(B_addrs, device='cuda')
    d_c_ptrs = torch.tensor(C_addrs, device='cuda')
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device='cuda')
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device='cuda')

    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel_expert[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size)
    return group_C

# ==================== BENCHMARK ====================

print("="*80)
print("GROUPED GEMM: 3-WAY COMPARISON")
print("="*80)
print("\n1. PyTorch Baseline - Loop with separate kernel launches")
print("2. Expert Triton - Official tutorial (optimized for H100)")
print("3. LLM Triton - Generated with RTX 3090 context")
print("-"*80)

test_configs = [
    # Original small sizes (for comparison)
    {'name': '8 GEMMs (1024×1024)', 'group_size': 8, 'sizes': [(1024, 1024, 1024)] * 8},
    # Larger sizes that should favor Triton
    {'name': '16 GEMMs (2048×2048)', 'group_size': 16, 'sizes': [(2048, 2048, 2048)] * 16},
    {'name': '32 GEMMs (2048×2048)', 'group_size': 32, 'sizes': [(2048, 2048, 2048)] * 32},
    {'name': '64 GEMMs (1024×1024)', 'group_size': 64, 'sizes': [(1024, 1024, 1024)] * 64},
]

print(f"\n{'Config':<25} {'PyTorch':<15} {'Expert':<15} {'LLM':<15} {'LLM/PyTorch':<15} {'LLM/Expert':<15}")
print("-"*100)

results = []

for config in test_configs:
    name, group_size, sizes = config['name'], config['group_size'], config['sizes']

    group_A, group_B = [], []
    for M, N, K in sizes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)

    # Warmup
    for _ in range(3):
        _ = grouped_gemm_baseline(group_A, group_B)
        _ = grouped_gemm_expert(group_A, group_B)
        _ = grouped_gemm_llm(group_A, group_B)
    torch.cuda.synchronize()

    # Benchmark (10 iterations for larger problem sizes)
    start = time.perf_counter()
    for _ in range(10):
        c_pytorch = grouped_gemm_baseline(group_A, group_B)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 10 * 1000

    start = time.perf_counter()
    for _ in range(10):
        c_expert = grouped_gemm_expert(group_A, group_B)
    torch.cuda.synchronize()
    expert_time = (time.perf_counter() - start) / 10 * 1000

    start = time.perf_counter()
    for _ in range(10):
        c_llm = grouped_gemm_llm(group_A, group_B)
    torch.cuda.synchronize()
    llm_time = (time.perf_counter() - start) / 10 * 1000

    # Verify (relaxed tolerance for float16)
    for i in range(group_size):
        max_diff_expert = torch.max(torch.abs(c_pytorch[i] - c_expert[i])).item()
        max_diff_llm = torch.max(torch.abs(c_pytorch[i] - c_llm[i])).item()
        assert max_diff_expert < 1.0, f"Expert GEMM {i}: Max diff: {max_diff_expert}"
        assert max_diff_llm < 1.0, f"LLM GEMM {i}: Max diff: {max_diff_llm}"

    llm_vs_pytorch = pytorch_time / llm_time
    llm_vs_expert = expert_time / llm_time

    print(f"{name:<25} {pytorch_time:>10.4f} ms  {expert_time:>10.4f} ms  {llm_time:>10.4f} ms  {llm_vs_pytorch:>13.2f}x  {llm_vs_expert:>13.2f}x")

    results.append({
        'name': name,
        'pytorch_time': pytorch_time,
        'expert_time': expert_time,
        'llm_time': llm_time,
        'llm_vs_pytorch': llm_vs_pytorch,
        'llm_vs_expert': llm_vs_expert
    })

print("="*100)
print("\n📊 SUMMARY")
print("-"*100)

import numpy as np
avg_vs_pytorch = np.mean([r['llm_vs_pytorch'] for r in results])
avg_vs_expert = np.mean([r['llm_vs_expert'] for r in results])

print(f"\nLLM vs PyTorch baseline: {avg_vs_pytorch:.2f}x")
print(f"LLM vs Expert Triton:    {avg_vs_expert:.2f}x")

if avg_vs_pytorch >= 1.5:
    print("\n✅ EXCELLENT - LLM Triton significantly faster than PyTorch")
elif avg_vs_pytorch >= 1.1:
    print("\n✓✓ GOOD - LLM Triton moderately faster than PyTorch")
elif avg_vs_pytorch >= 0.9:
    print("\n✓  ADEQUATE - LLM Triton matches PyTorch")
else:
    print("\n⚠  LLM Triton slower than PyTorch baseline")

if avg_vs_expert >= 1.1:
    print("✅ LLM outperforms Expert Triton!")
elif avg_vs_expert >= 0.9:
    print("✅ LLM matches Expert Triton performance")
else:
    print(f"⚠  LLM is {1/avg_vs_expert:.2f}x slower than Expert")

print("\nKey Insight:")
print("  With proper hardware context, LLM-generated Triton may be able to")
print("  match or outperform generic expert implementations by tuning for")
print("  specific GPU architecture and problem sizes.")
