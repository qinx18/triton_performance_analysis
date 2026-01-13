import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    # Get block offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (0 to N-2, since we process i < N-1)
    mask = indices < (N - 1)
    
    # Load arrays for current indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + indices, b_new_vals, mask=mask)

@triton.jit
def s244_epilogue_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    N,
):
    # Execute S2 only at the last iteration (i = N-2)
    i = N - 2
    if i >= 0:
        b_val = tl.load(b_ptr + i)
        a_next_val = tl.load(a_ptr + i + 1)
        d_val = tl.load(d_ptr + i)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_next_new_val = b_val + a_next_val * d_val
        tl.store(a_ptr + i + 1, a_next_new_val)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    
    # We need to process indices 0 to N-2 (since loop condition is i < N-1)
    n_elements = N - 1
    
    if n_elements > 0:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # Main kernel for S0 and S1
        s244_kernel[grid](
            a, b, c, d,
            N,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Epilogue kernel for S2 at last iteration only
        s244_epilogue_kernel[(1,)](
            a, b, c, d,
            N,
        )