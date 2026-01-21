import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the loop sequentially since j depends on conditional increments
    # We'll process one iteration at a time to maintain the correct indexing behavior
    
    pid = tl.program_id(0)
    
    # Each program processes BLOCK_SIZE iterations
    start_i = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = start_i + offsets
    mask = i_offsets < n_half
    
    # Load data for this block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Process each element in the block sequentially
    for idx in range(BLOCK_SIZE):
        i = start_i + idx
        
        # Check bounds
        valid = i < n_half
        if not valid:
            return
            
        j = 2 * i  # j starts at -1, then increments: j becomes 0, 2, 4, ... for i = 0, 1, 2, ...
        
        # Load individual values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        val1 = b_val + d_val * e_val
        tl.store(a_ptr + j, val1)
        
        # Conditional assignment
        if c_val > 0.0:
            j = j + 1
            val2 = c_val + d_val * e_val
            tl.store(a_ptr + j, val2)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e, n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )