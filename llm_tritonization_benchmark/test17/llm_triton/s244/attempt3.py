import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for main loop (i < n_elements - 1)
    main_mask = offsets < (n_elements - 1)
    
    # Load values for main loop
    a_vals = tl.load(a_ptr + offsets, mask=main_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=main_mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=main_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=main_mask, other=0.0)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_new = b_vals + c_vals * d_vals
    
    # S1: b[i] = c[i] + b[i]
    b_new = c_vals + b_vals
    
    # Store results for S0 and S1
    tl.store(a_ptr + offsets, a_new, mask=main_mask)
    tl.store(b_ptr + offsets, b_new, mask=main_mask)

@triton.jit
def s244_epilogue_kernel(
    a_ptr, b_ptr, d_ptr,
    n_elements,
):
    # Execute S2 only for the last iteration (i = n_elements - 2)
    i = n_elements - 2
    if i >= 0:
        # Load a[i+1] and b[i] and d[i] for the last iteration
        a_next = tl.load(a_ptr + i + 1)
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_next_new = b_val + a_next * d_val
        
        # Store result
        tl.store(a_ptr + i + 1, a_next_new)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    if n_elements <= 1:
        return a, b, c, d
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements - 1, BLOCK_SIZE)
    
    # Main loop - execute S0 and S1 for all iterations
    s244_kernel[(grid_size,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Epilogue - execute S2 only for the last iteration
    s244_epilogue_kernel[(1,)](
        a, b, d,
        n_elements,
    )
    
    return a, b, c, d