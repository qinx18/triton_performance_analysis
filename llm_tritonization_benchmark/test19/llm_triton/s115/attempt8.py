import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Only process i values where i > j
    actual_i = i_indices + j + 1
    mask = actual_i < N
    
    # Load a[j]
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][actual_i]
    aa_offsets = j * N + actual_i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[actual_i]
    a_vals = tl.load(a_ptr + actual_i, mask=mask)
    
    # Compute a[actual_i] -= aa[j][actual_i] * a[j]
    result = a_vals - aa_vals * a_j
    
    # Store back to a[actual_i]
    tl.store(a_ptr + actual_i, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(N):
        remaining = N - j - 1
        if remaining > 0:
            grid = (triton.cdiv(remaining, BLOCK_SIZE),)
            s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)