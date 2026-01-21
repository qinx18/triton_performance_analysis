import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(block_start, n, tl.num_programs(0) * BLOCK_SIZE):
        current_offsets = i + offsets
        mask = current_offsets < n
        
        # Load b[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # Load a[j] where j = i + 1
        j_offsets = current_offsets + 1
        j_mask = j_offsets < (n + 1)  # Allow reading one past for a[j]
        a_j_vals = tl.load(a_ptr + j_offsets, mask=j_mask)
        
        # Compute a[i] = a[j] + b[i]
        result = a_j_vals + b_vals
        
        # Store to a[i]
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s121_triton(a, b):
    n = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n, BLOCK_SIZE)