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
    mask = i_indices < N
    valid_mask = mask & (i_indices > j)
    
    # Load a[j]
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i]
    aa_offsets = j * N + i_indices
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=valid_mask, other=0.0)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + i_indices, mask=valid_mask, other=0.0)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_indices, result, mask=valid_mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(N):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)