import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = (i_offsets < n) & (i_offsets > j)
    
    # Load a[j] (scalar broadcast)
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i] values
    aa_offsets = j * n + i_offsets
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_vals - aa_vals * a_j
    
    # Store back to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s115_triton(a, aa):
    n = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(n):
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, n, BLOCK_SIZE)