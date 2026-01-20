import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets + j + 1
    
    mask = i_indices < N
    
    # Load a[j]
    a_j = tl.load(a_ptr + j)
    
    # Load aa[j][i_indices]
    aa_offsets = j * N + i_indices
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load a[i_indices]
    a_vals = tl.load(a_ptr + i_indices, mask=mask)
    
    # Compute a[i] -= aa[j][i] * a[j]
    result = a_vals - aa_vals * a_j
    
    # Store back to a[i_indices]
    tl.store(a_ptr + i_indices, result, mask=mask)

def s115_triton(a, aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(N):
        remaining = N - j - 1
        if remaining <= 0:
            continue
            
        grid = (triton.cdiv(remaining, BLOCK_SIZE),)
        s115_kernel[grid](a, aa, j, N, BLOCK_SIZE=BLOCK_SIZE)