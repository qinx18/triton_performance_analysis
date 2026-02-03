import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # For each position i, we need b[im1] where im1 is the previous index
        # im1 = LEN_1D-1 initially, then im1 = i from previous iteration
        # So: im1[i] = (i-1) for i > 0, and im1[0] = LEN_1D-1
        
        # Calculate im1 indices for current block
        im1_offsets = tl.where(current_offsets == 0, n_elements - 1, current_offsets - 1)
        
        # Load b[i] and b[im1]
        b_i = tl.load(b_ptr + current_offsets, mask=mask)
        b_im1 = tl.load(b_ptr + im1_offsets, mask=mask)
        
        # Calculate a[i] = (b[i] + b[im1]) * 0.5
        result = (b_i + b_im1) * 0.5
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s291_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        a, b, n_elements, BLOCK_SIZE
    )