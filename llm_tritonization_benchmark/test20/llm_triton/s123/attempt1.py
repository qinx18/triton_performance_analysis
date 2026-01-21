import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the complex induction variable pattern
    # Each thread block handles a portion of the input sequentially
    pid = tl.program_id(0)
    
    # Process elements sequentially within each block
    block_start = pid * BLOCK_SIZE
    
    if block_start >= n:
        return
    
    # Calculate how many elements this block will process
    elements_in_block = min(BLOCK_SIZE, n - block_start)
    
    # Starting j value for this block - need to calculate based on previous blocks
    j_start = block_start  # Base j value
    
    # Add extra increments from previous blocks due to positive c values
    for prev_i in range(block_start):
        c_val = tl.load(c_ptr + prev_i)
        if c_val > 0.0:
            j_start += 1
    
    current_j = j_start
    
    # Process elements in this block sequentially
    for offset in range(elements_in_block):
        i = block_start + offset
        if i >= n:
            break
            
        # Load values for current i
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + current_j, result1)
        current_j += 1
        
        # Conditional assignment
        if c_val > 0.0:
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + current_j, result2)
            current_j += 1

def s123_triton(a, b, c, d, e):
    n = b.shape[0] // 2  # LEN_1D/2 from the original loop
    
    # Use small block size since we need sequential processing
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )