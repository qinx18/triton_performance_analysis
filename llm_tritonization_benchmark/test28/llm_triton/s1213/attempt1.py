import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = (idx >= 1) & (idx < N - 1)
    
    # Load data for prologue (i=1 case)
    if pid == 0:
        # a[1] = b[0] + c[1]
        b_0 = tl.load(b_ptr)
        c_1 = tl.load(c_ptr + 1)
        a_1 = b_0 + c_1
        tl.store(a_ptr + 1, a_1)
    
    # Main parallel computation for i in range(1, N-2)
    main_mask = (idx >= 1) & (idx < N - 2)
    
    if tl.any(main_mask):
        # Step 1: Compute b[i] = a_copy[i+1] * d[i]
        a_copy_vals = tl.load(a_copy_ptr + idx + 1, mask=main_mask)
        d_vals = tl.load(d_ptr + idx, mask=main_mask)
        b_vals = a_copy_vals * d_vals
        tl.store(b_ptr + idx, b_vals, mask=main_mask)
        
        # Step 2: Compute a[i+1] = b[i] + c[i+1]
        c_vals = tl.load(c_ptr + idx + 1, mask=main_mask)
        a_vals = b_vals + c_vals
        tl.store(a_ptr + idx + 1, a_vals, mask=main_mask)
    
    # Epilogue: b[N-2] = a_copy[N-1] * d[N-2]
    if pid == (triton.cdiv(N - 2, BLOCK_SIZE) - 1):
        last_idx = N - 2
        if last_idx >= 1:
            a_copy_last = tl.load(a_copy_ptr + N - 1)
            d_last = tl.load(d_ptr + last_idx)
            b_last = a_copy_last * d_last
            tl.store(b_ptr + last_idx, b_last)

def s1213_triton(a, b, c, d):
    N = a.shape[0]
    
    # Create read-only copy to handle WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1213_kernel[grid](
        a, b, c, d, a_copy,
        N,
        BLOCK_SIZE,
    )