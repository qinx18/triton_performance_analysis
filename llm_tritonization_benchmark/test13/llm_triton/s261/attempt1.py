import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets once
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Prologue: First consumer iteration (i=1, uses original c[0])
    if pid == 0:
        # Load original c[0] for first iteration
        c_0 = tl.load(c_ptr)  # c[0]
        
        # Compute a[1] using original c[0]
        a_1 = tl.load(a_ptr + 1)
        b_1 = tl.load(b_ptr + 1)
        t = a_1 + b_1
        a_1_new = t + c_0
        tl.store(a_ptr + 1, a_1_new)
    
    # Main loop: Reordered - producer then shifted consumer
    # Process elements [1, n_elements-2] in parallel
    block_start = pid * BLOCK_SIZE + 1
    current_offsets = block_start + offsets
    mask = (current_offsets >= 1) & (current_offsets < n_elements - 1)
    
    # Step 1: Compute c[i] (producer)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    t_vals = c_vals * d_vals
    tl.store(c_ptr + current_offsets, t_vals, mask=mask)
    
    # Step 2: Use c[i] for next position a[i+1] (shifted consumer)
    next_offsets = current_offsets + 1
    next_mask = (next_offsets >= 2) & (next_offsets < n_elements)
    
    if tl.sum(next_mask.to(tl.int32)) > 0:
        a_next = tl.load(a_ptr + next_offsets, mask=next_mask)
        b_next = tl.load(b_ptr + next_offsets, mask=next_mask)
        t_next = a_next + b_next
        a_next_new = t_next + t_vals  # Use computed c[i] for a[i+1]
        tl.store(a_ptr + next_offsets, a_next_new, mask=next_mask)

@triton.jit
def s261_epilogue_kernel(c_ptr, d_ptr, n_elements):
    # Epilogue: Last producer iteration c[n_elements-1]
    if n_elements > 1:
        last_idx = n_elements - 1
        c_last = tl.load(c_ptr + last_idx)
        d_last = tl.load(d_ptr + last_idx)
        t_last = c_last * d_last
        tl.store(c_ptr + last_idx, t_last)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    if n_elements <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Launch main kernel for reordered computation
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Launch epilogue kernel for last c[n_elements-1] computation
    s261_epilogue_kernel[(1,)](c, d, n_elements)