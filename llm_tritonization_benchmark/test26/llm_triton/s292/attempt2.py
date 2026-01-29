import triton
import triton.language as tl

@triton.jit
def s292_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # This kernel processes sequentially due to dependencies
    # Each thread block handles the entire array sequentially
    if pid == 0:
        # Sequential processing with wrap-around variables
        for i in range(N):
            # Calculate im1 and im2 based on iteration pattern
            if i == 0:
                im1 = N - 1
                im2 = N - 2
            elif i == 1:
                im1 = 0
                im2 = N - 1
            else:
                im1 = i - 1
                im2 = i - 2
            
            # Load values
            b_i = tl.load(b_ptr + i)
            b_im1 = tl.load(b_ptr + im1)
            b_im2 = tl.load(b_ptr + im2)
            
            # Compute result
            result = (b_i + b_im1 + b_im2) * 0.333
            
            # Store result
            tl.store(a_ptr + i, result)

def s292_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single block since computation is sequential
    grid = (1,)
    
    s292_kernel[grid](
        a, b, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a