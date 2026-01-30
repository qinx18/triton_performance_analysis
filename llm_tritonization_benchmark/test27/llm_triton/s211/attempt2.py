import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Prologue: First consumer uses original b[0]
    if tl.program_id(0) == 0:
        c_1 = tl.load(c_ptr + 1)
        d_1 = tl.load(d_ptr + 1)
        b_0 = tl.load(b_copy_ptr + 0)
        a_1 = b_0 + c_1 * d_1
        tl.store(a_ptr + 1, a_1)
    
    # Main loop: parallelizable after reordering
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Adjust range for main loop: i from 1 to N-3
    main_offsets = 1 + offsets
    mask = main_offsets < N - 2
    
    # Load data for main loop
    e_vals = tl.load(e_ptr + main_offsets, mask=mask)
    d_vals = tl.load(d_ptr + main_offsets, mask=mask)
    b_next_vals = tl.load(b_copy_ptr + main_offsets + 1, mask=mask)
    c_next_vals = tl.load(c_ptr + main_offsets + 1, mask=mask)
    d_next_vals = tl.load(d_ptr + main_offsets + 1, mask=mask)
    
    # Compute producer: b[i] = b_copy[i+1] - e[i] * d[i]
    b_vals = b_next_vals - e_vals * d_vals
    tl.store(b_ptr + main_offsets, b_vals, mask=mask)
    
    # Compute shifted consumer: a[i+1] = b[i] + c[i+1] * d[i+1]
    a_next_vals = b_vals + c_next_vals * d_next_vals
    tl.store(a_ptr + main_offsets + 1, a_next_vals, mask=mask)
    
    # Epilogue: Last producer has no consumer
    epilogue_idx = N - 2
    if pid * BLOCK_SIZE <= epilogue_idx < (pid + 1) * BLOCK_SIZE:
        e_last = tl.load(e_ptr + epilogue_idx)
        d_last = tl.load(d_ptr + epilogue_idx)
        b_last_next = tl.load(b_copy_ptr + epilogue_idx + 1)
        b_last = b_last_next - e_last * d_last
        tl.store(b_ptr + epilogue_idx, b_last)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )