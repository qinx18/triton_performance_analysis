import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel requires sequential processing due to dependencies
    # Process one element per thread block to maintain order
    block_id = tl.program_id(0)
    
    if block_id == 0:
        # Only one block processes the entire array sequentially
        im1 = N - 1
        im2 = N - 2
        
        for i in range(N):
            # Load b[i], b[im1], b[im2]
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute result
            result = (b_i + b_im1 + b_im2) * 0.333
            
            # Store result
            tl.store(a_ptr + i, result)
            
            # Update indices
            im2 = im1
            im1 = i

def s292_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since sequential processing is required
    grid = (1,)
    s292_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)