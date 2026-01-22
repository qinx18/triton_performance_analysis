import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    # Load a[j] where j = i + 1, so a[offsets + 1]
    a_next_offsets = offsets + 1
    a_next_mask = a_next_offsets < (n + 1)  # a has n+1 elements, accessing up to index n
    a_next = tl.load(a_ptr + a_next_offsets, mask=a_next_mask & mask)
    
    # Load b[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_next + b_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s121_triton(a, b):
    n = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-1, so n = LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)