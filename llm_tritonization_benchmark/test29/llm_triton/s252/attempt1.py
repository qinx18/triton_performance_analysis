import triton
import triton.language as tl

@triton.jit
def s252_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process each element in the block sequentially to maintain dependencies
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            # Load b[idx] and c[idx]
            b_val = tl.load(b_ptr + idx)
            c_val = tl.load(c_ptr + idx)
            
            # Compute s = b[idx] * c[idx]
            s = b_val * c_val
            
            # Compute t based on previous iteration
            if idx == 0:
                t = 0.0
            else:
                # t = previous s = b[idx-1] * c[idx-1]
                prev_b = tl.load(b_ptr + idx - 1)
                prev_c = tl.load(c_ptr + idx - 1)
                t = prev_b * prev_c
            
            # Compute a[idx] = s + t
            a_val = s + t
            
            # Store result
            tl.store(a_ptr + idx, a_val)

def s252_triton(a, b, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s252_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )