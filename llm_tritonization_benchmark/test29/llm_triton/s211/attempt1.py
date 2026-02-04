import triton
import triton.language as tl

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Prologue: First consumer uses original b[0]
    pid = tl.program_id(0)
    if pid == 0:
        b_val = tl.load(b_copy_ptr)
        c_val = tl.load(c_ptr + 1)
        d_val = tl.load(d_ptr + 1)
        a_val = b_val + c_val * d_val
        tl.store(a_ptr + 1, a_val)
    
    # Main parallel loop: each thread handles one value of i
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    i_offsets = block_start + offsets + 1
    
    mask = i_offsets < (N - 2)
    
    # Step 1: Compute b[i] (producer)
    b_read_vals = tl.load(b_copy_ptr + i_offsets + 1, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    b_new_vals = b_read_vals - e_vals * d_vals
    tl.store(b_ptr + i_offsets, b_new_vals, mask=mask)
    
    # Step 2: Compute a[i+1] using b[i] just computed
    c_vals = tl.load(c_ptr + i_offsets + 1, mask=mask)
    d_vals_next = tl.load(d_ptr + i_offsets + 1, mask=mask)
    a_new_vals = b_new_vals + c_vals * d_vals_next
    tl.store(a_ptr + i_offsets + 1, a_new_vals, mask=mask)

@triton.jit
def s211_epilogue_kernel(b_ptr, b_copy_ptr, e_ptr, d_ptr, N):
    # Epilogue: Last producer (no corresponding consumer)
    b_val = tl.load(b_copy_ptr + N - 1)
    e_val = tl.load(e_ptr + N - 2)
    d_val = tl.load(d_ptr + N - 2)
    b_new_val = b_val - e_val * d_val
    tl.store(b_ptr + N - 2, b_new_val)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(N - 3, BLOCK_SIZE)
    
    # Main parallel kernel
    if num_blocks > 0:
        s211_kernel[(num_blocks,)](
            a, b, b_copy, c, d, e, N, BLOCK_SIZE
        )
    else:
        # Handle prologue only for very small arrays
        if N > 2:
            b_val = b_copy[0]
            c_val = c[1]
            d_val = d[1]
            a[1] = b_val + c_val * d_val
    
    # Epilogue kernel
    if N > 2:
        s211_epilogue_kernel[(1,)](b, b_copy, e, d, N)