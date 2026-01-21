import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program handles one block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= N:
            break
            
        # Calculate wrap-around indices
        if idx == 0:
            im1 = N - 1
            im2 = N - 2
        elif idx == 1:
            im1 = 0
            im2 = N - 1
        else:
            im1 = idx - 1
            im2 = idx - 2
            
        # Load values
        b_i = tl.load(b_ptr + idx)
        b_im1 = tl.load(b_ptr + im1)
        b_im2 = tl.load(b_ptr + im2)
        
        # Compute and store
        result = (b_i + b_im1 + b_im2) * 0.333
        tl.store(a_ptr + idx, result)

def s292_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s292_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)