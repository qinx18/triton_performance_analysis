import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks
    for start in range(0, N, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < N
        
        # Load b values for current block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # For each element in block, compute with proper im1, im2 dependencies
        for local_i in range(BLOCK_SIZE):
            global_i = start + local_i
            if global_i >= N:
                break
                
            # Calculate im1 and im2 based on current iteration
            if global_i == 0:
                im1 = N - 1
                im2 = N - 2
            elif global_i == 1:
                im1 = 0
                im2 = N - 1
            else:
                im1 = global_i - 1
                im2 = global_i - 2
            
            # Load required b values
            b_i = tl.load(b_ptr + global_i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute result
            result = (b_i + b_im1 + b_im2) * 0.333
            
            # Store result
            tl.store(a_ptr + global_i, result)

def s292_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s292_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a