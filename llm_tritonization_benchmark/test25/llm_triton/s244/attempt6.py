import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Process the main loop (i = 0 to N-2)
    for block_start in range(0, N - 1, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N - 1
        
        # Load values
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        d_vals = tl.load(d_ptr + offsets, mask=mask)
        a_next_vals = tl.load(a_ptr + offsets + 1, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_new = b_vals + c_vals * d_vals
        tl.store(a_ptr + offsets, a_new, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + offsets, b_new, mask=mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i] (only for last iteration)
        is_last_iter = offsets == (N - 2)
        last_mask = mask & is_last_iter
        if tl.any(last_mask):
            a_next_new = b_new + a_next_vals * d_vals
            tl.store(a_ptr + offsets + 1, a_next_new, mask=last_mask)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (1,)
    s244_kernel[grid](a, b, c, d, N, BLOCK_SIZE)