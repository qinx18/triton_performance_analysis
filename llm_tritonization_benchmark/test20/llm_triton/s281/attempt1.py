import triton
import triton.language as tl
import torch

@triton.jit
def s281_expand_x_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    x_val = 0.0
    for i in range(n):
        a_val = tl.load(a_ptr + (n - 1 - i))
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        x_val = a_val + b_val * c_val
        tl.store(x_ptr + i, x_val)

@triton.jit
def s281_phase1_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, x_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = (indices < threshold) & (indices >= 0)
    
    # Load expanded x values
    x_vals = tl.load(x_ptr + indices, mask=mask)
    
    # Store results
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

@triton.jit
def s281_phase2_kernel(a_ptr, b_ptr, c_ptr, x_ptr, n, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + threshold
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = (indices < n) & (indices >= threshold)
    
    # Load expanded x values
    x_vals = tl.load(x_ptr + indices, mask=mask)
    
    # Store results
    tl.store(a_ptr + indices, x_vals - 1.0, mask=mask)
    tl.store(b_ptr + indices, x_vals, mask=mask)

def s281_triton(a, b, c, x):
    n = a.shape[0]
    threshold = n // 2
    BLOCK_SIZE = 256
    
    # Create expanded x array
    x_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar x
    grid = (1,)
    s281_expand_x_kernel[grid](a, b, c, x_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Clone array for phase 1 reads
    a_copy = a.clone()
    
    # Phase 1: Process first half (i = 0 to threshold-1)
    grid1 = (triton.cdiv(threshold, BLOCK_SIZE),)
    s281_phase1_kernel[grid1](a, a_copy, b, c, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Process second half (i = threshold to n-1)
    grid2 = (triton.cdiv(n - threshold, BLOCK_SIZE),)
    s281_phase2_kernel[grid2](a, b, c, x_expanded, n, threshold, BLOCK_SIZE=BLOCK_SIZE)