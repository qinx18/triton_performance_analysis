import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Main computation is for i from 1 to n-1
    valid_mask = (idx >= 1) & (idx < n)
    
    # Load arrays for current indices (i)
    a_vals = tl.load(a_ptr + idx, mask=valid_mask, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=valid_mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=valid_mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=valid_mask, other=0.0)
    
    # Load c[i-1] values (need mask for idx-1 >= 0)
    c_prev_mask = (idx >= 1) & (idx < n)
    c_prev_vals = tl.load(c_ptr + (idx - 1), mask=c_prev_mask, other=0.0)
    
    # Apply the C algorithm:
    # t = a[i] + b[i];
    # a[i] = t + c[i-1];
    # t = c[i] * d[i];
    # c[i] = t;
    
    t1 = a_vals + b_vals
    new_a = t1 + c_prev_vals
    
    t2 = c_vals * d_vals
    new_c = t2
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=valid_mask)
    tl.store(c_ptr + idx, new_c, mask=valid_mask)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    
    # For the dependency a[i] depends on c[i-1], we need to process sequentially
    # However, we can still process in blocks with careful handling
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # Process in sequential blocks to maintain dependency
    for block_idx in range(grid_size):
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n)
        
        if block_start >= n - 1:
            break
            
        # Adjust block_start to ensure we start from at least index 1
        actual_start = max(block_start, 1)
        
        if actual_start >= block_end:
            continue
            
        # Launch kernel for this block
        s261_kernel[(1,)](
            a, b, c, d, n,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return a, c