import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel_main(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid indices (0 to n_elements-2)
    mask = idx < (n_elements - 1)
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    tl.store(b_ptr + idx, b_new_vals, mask=mask)

@triton.jit
def s244_kernel_epilogue(
    a_ptr, b_ptr, c_ptr, d_ptr,
    last_idx,
    BLOCK_SIZE: tl.constexpr,
):
    # Only execute for the last iteration (i = N-2)
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    # Load values at last iteration index
    b_val = tl.load(b_ptr + last_idx)
    d_val = tl.load(d_ptr + last_idx)
    a_next_val = tl.load(a_ptr + last_idx + 1)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i]
    a_next_new_val = b_val + a_next_val * d_val
    tl.store(a_ptr + last_idx + 1, a_next_new_val)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    
    # Handle edge case
    if N <= 1:
        return
    
    BLOCK_SIZE = 256
    
    # Main kernel for S0 and S1
    grid_main = (triton.cdiv(N - 1, BLOCK_SIZE),)
    s244_kernel_main[grid_main](
        a, b, c, d,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Epilogue kernel for S2 at last iteration only
    grid_epilogue = (1,)
    s244_kernel_epilogue[grid_epilogue](
        a, b, c, d,
        N - 2,
        BLOCK_SIZE=BLOCK_SIZE,
    )