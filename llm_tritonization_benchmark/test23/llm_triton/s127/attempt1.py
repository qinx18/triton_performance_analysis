import triton
import triton.language as tl

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process pairs of outputs per input
    for i in range(BLOCK_SIZE):
        input_idx = block_start + i
        if input_idx >= n // 2:
            break
            
        # Load input values
        b_val = tl.load(b_ptr + input_idx)
        c_val = tl.load(c_ptr + input_idx) 
        d_val = tl.load(d_ptr + input_idx)
        e_val = tl.load(e_ptr + input_idx)
        
        # Calculate j indices (j starts at -1, increments twice per i)
        j1 = 2 * input_idx
        j2 = 2 * input_idx + 1
        
        # Store results
        if j1 < n:
            result1 = b_val + c_val * d_val
            tl.store(a_ptr + j1, result1)
            
        if j2 < n:
            result2 = b_val + d_val * e_val
            tl.store(a_ptr + j2, result2)

def s127_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n // 2, BLOCK_SIZE),)
    
    s127_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)