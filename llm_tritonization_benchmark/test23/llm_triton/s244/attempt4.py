import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load values for current iteration
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

@triton.jit
def s244_epilogue_kernel(a_ptr, b_ptr, d_ptr, n):
    if n >= 2:
        i = n - 2
        # Load the updated b[i] value and existing a[i+1] value
        b_i = tl.load(b_ptr + i)
        a_i_plus_1 = tl.load(a_ptr + i + 1)
        d_i = tl.load(d_ptr + i)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_new = b_i + a_i_plus_1 * d_i
        tl.store(a_ptr + i + 1, a_new)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    if n > 1:
        # Execute main loop for S0 and S1
        grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
        s244_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)
        
        # Execute epilogue for S2 (only last iteration)
        s244_epilogue_kernel[(1,)](a, b, d, n)