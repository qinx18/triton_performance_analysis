import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values
        a_val = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_val = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_val = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_val = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_val = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check conditions
        a_condition = a_val >= 0.0
        b_condition = b_val >= 0.0
        
        # If a[i] >= 0, skip both operations (goto L20)
        # If b[i] >= 0 but a[i] < 0, skip first operation but do second (goto L30)
        # If both a[i] < 0 and b[i] < 0, do first operation and then second
        
        # First operation: a[i] += c[i] * d[i] (only if a[i] < 0 and b[i] < 0)
        do_first_op = ~a_condition & ~b_condition
        a_new = tl.where(do_first_op, a_val + c_val * d_val, a_val)
        
        # Second operation: b[i+1] = c[i] + d[i] * e[i] (if a[i] < 0)
        do_second_op = ~a_condition
        b_new_val = c_val + d_val * e_val
        
        # Store updated a values
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Store b[i+1] values (need to handle boundary)
        next_offsets = current_offsets + 1
        next_mask = (next_offsets < (n + 1)) & mask & do_second_op
        tl.store(b_ptr + next_offsets, b_new_val, mask=next_mask)

def s277_triton(a, b, c, d, e):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    BLOCK_SIZE = 256
    
    # Launch kernel with single block since we're processing sequentially
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )