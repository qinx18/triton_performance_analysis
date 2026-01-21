import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # Each program processes one element to maintain sequential j behavior
    pid = tl.program_id(0)
    
    # Process multiple elements per program to improve efficiency
    start_i = pid * BLOCK_SIZE
    
    # Process each element in the block
    for idx in range(BLOCK_SIZE):
        i = start_i + idx
        
        # Check bounds
        if i >= n_half:
            break
            
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
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e, n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )