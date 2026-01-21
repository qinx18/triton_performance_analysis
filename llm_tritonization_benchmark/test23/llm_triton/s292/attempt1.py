import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel needs to handle the sequential dependency
    # We'll process one element at a time to maintain correctness
    pid = tl.program_id(0)
    
    if pid == 0:  # Only one program handles the entire computation
        # Initialize im1 and im2
        im1 = n - 1
        im2 = n - 2
        
        # Process each element sequentially
        for i in range(n):
            # Load b[i], b[im1], b[im2]
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute result
            result = (b_i + b_im1 + b_im2) * 0.333
            
            # Store to a[i]
            tl.store(a_ptr + i, result)
            
            # Update im1 and im2 for next iteration
            im2 = im1
            im1 = i

def s292_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with only one program since we need sequential processing
    grid = (1,)
    
    s292_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )