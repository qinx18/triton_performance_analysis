import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S1 for all iterations (i = 0 to n-2)
    main_offsets = block_start + offsets
    main_mask = (main_offsets < (n - 1))
    
    # Load data for main computation
    b_vals = tl.load(b_ptr + main_offsets, mask=main_mask, other=0.0)
    c_vals = tl.load(c_ptr + main_offsets, mask=main_mask, other=0.0)
    
    # S1: a[i] = b[i] + c[i] for all valid iterations
    result_s1 = b_vals + c_vals
    tl.store(a_ptr + main_offsets, result_s1, mask=main_mask)
    
    # Epilogue - execute S0 only at last iteration (i = n-2)
    # S0: a[i+1] = b[i] + e[i] where i = n-2, so a[n-1] = b[n-2] + e[n-2]
    epilogue_i = n - 2
    epilogue_mask = (main_offsets == epilogue_i)
    
    if tl.any(epilogue_mask):
        e_vals = tl.load(e_ptr + main_offsets, mask=epilogue_mask, other=0.0)
        result_s0 = b_vals + e_vals
        # Store at a[i+1] = a[epilogue_i + 1]
        epilogue_store_offsets = main_offsets + 1
        epilogue_store_mask = epilogue_mask & (epilogue_store_offsets < n)
        tl.store(a_ptr + epilogue_store_offsets, result_s0, mask=epilogue_store_mask)

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )