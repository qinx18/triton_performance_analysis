import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially block by block
    for block_start in range(1, n, BLOCK_SIZE):
        idx = block_start + offsets
        valid_mask = (idx >= 1) & (idx < n)
        
        if not tl.sum(valid_mask.to(tl.int32)):
            continue
            
        # Load current values
        a_vals = tl.load(a_ptr + idx, mask=valid_mask, other=0.0)
        b_vals = tl.load(b_ptr + idx, mask=valid_mask, other=0.0)
        c_vals = tl.load(c_ptr + idx, mask=valid_mask, other=0.0)
        d_vals = tl.load(d_ptr + idx, mask=valid_mask, other=0.0)
        
        # Load c[i-1] values
        c_prev_idx = idx - 1
        c_prev_mask = (c_prev_idx >= 0) & valid_mask
        c_prev_vals = tl.load(c_ptr + c_prev_idx, mask=c_prev_mask, other=0.0)
        
        # Compute: t = a[i] + b[i]; a[i] = t + c[i-1]
        t1 = a_vals + b_vals
        new_a = t1 + c_prev_vals
        
        # Compute: t = c[i] * d[i]; c[i] = t
        new_c = c_vals * d_vals
        
        # Store results
        tl.store(a_ptr + idx, new_a, mask=valid_mask)
        tl.store(c_ptr + idx, new_c, mask=valid_mask)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    
    # Process element by element to maintain dependency
    for i in range(1, n):
        t = a[i] + b[i]
        a[i] = t + c[i-1]
        t = c[i] * d[i]
        c[i] = t
    
    return a, c