import triton
import triton.language as tl
import torch

@triton.jit
def s292_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize result vector
    results = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process each element in the block
    im1_offsets = tl.where(offsets == 0, N - 1, 
                  tl.where(offsets == 1, 0, offsets - 1))
    im2_offsets = tl.where(offsets == 0, N - 2,
                  tl.where(offsets == 1, N - 1, offsets - 2))
    
    # Load im1 and im2 values
    b_im1_vals = tl.load(b_ptr + im1_offsets, mask=mask, other=0.0)
    b_im2_vals = tl.load(b_ptr + im2_offsets, mask=mask, other=0.0)
    
    # Compute results
    results = (b_vals + b_im1_vals + b_im2_vals) * 0.333
    
    # Store results
    tl.store(a_ptr + offsets, results, mask=mask)

def s292_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s292_kernel[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)