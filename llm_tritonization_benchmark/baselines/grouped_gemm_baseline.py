"""
Grouped GEMM - Baseline Implementation
Source: Triton tutorial 08-grouped-gemm.py

Grouped GEMM: Execute multiple independent matrix multiplications in parallel.
This is a "matrix operation beyond BLAS" - standard BLAS doesn't have a single
call for batched/grouped independent GEMMs.

Use cases:
- Mixture-of-Experts (MoE) models
- Multi-task learning with different weight matrices
- Dynamic batching scenarios

Baseline approach: Loop over groups with separate kernel launches
"""

import torch

def grouped_gemm_baseline(group_A, group_B):
    """
    Naive baseline: Execute each GEMM separately in a Python loop

    This results in:
    - Multiple kernel launches (one per GEMM)
    - Poor GPU utilization due to launch overhead
    - No parallelism across different GEMMs

    Args:
        group_A: List of input matrices A_i of shape (M_i, K_i)
        group_B: List of input matrices B_i of shape (K_i, N_i)

    Returns:
        List of output matrices C_i of shape (M_i, N_i)
        where C_i = A_i @ B_i
    """
    assert len(group_A) == len(group_B), "Group sizes must match"

    # Simple loop - each matmul is a separate kernel launch
    group_C = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]

    return group_C


def grouped_gemm_manual_loop(group_A, group_B):
    """
    Alternative baseline: Explicit loop (more verbose but same performance)
    """
    assert len(group_A) == len(group_B), "Group sizes must match"

    group_C = []
    for a, b in zip(group_A, group_B):
        assert a.shape[1] == b.shape[0], f"Dimension mismatch: {a.shape} @ {b.shape}"
        c = torch.matmul(a, b)
        group_C.append(c)

    return group_C


def grouped_gemm_stacked(group_A, group_B):
    """
    Alternative: Try to stack and use batched matmul

    This only works if all matrices have the same dimensions.
    For variable-sized groups, this approach fails.

    Most real use cases (MoE, etc.) have variable sizes.
    """
    # Check if all matrices have same size
    shapes_A = [a.shape for a in group_A]
    shapes_B = [b.shape for b in group_B]

    if len(set(shapes_A)) == 1 and len(set(shapes_B)) == 1:
        # All same size - can use batched matmul
        stacked_A = torch.stack(group_A, dim=0)  # (batch, M, K)
        stacked_B = torch.stack(group_B, dim=0)  # (batch, K, N)
        stacked_C = torch.bmm(stacked_A, stacked_B)  # (batch, M, N)
        return list(torch.unbind(stacked_C, dim=0))
    else:
        # Variable sizes - fall back to loop
        return grouped_gemm_baseline(group_A, group_B)
