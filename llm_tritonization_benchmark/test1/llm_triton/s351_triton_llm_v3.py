import triton
import triton.language as tl
import torch

@triton.jit
def s351_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Load alpha once per block
    alpha = tl.load(c_ptr)
    
    # Handle unrolled loop (step by 5)
    for offset in range(0, BLOCK_SIZE, 5):
        idx_base = block_start + offset
        
        # Create masks for the 5 elements
        mask0 = idx_base < n_elements
        mask1 = (idx_base + 1) < n_elements
        mask2 = (idx_base + 2) < n_elements
        mask3 = (idx_base + 3) < n_elements
        mask4 = (idx_base + 4) < n_elements
        
        # Load elements from a and b
        a0 = tl.load(a_ptr + idx_base, mask=mask0, other=0.0)
        a1 = tl.load(a_ptr + idx_base + 1, mask=mask1, other=0.0)
        a2 = tl.load(a_ptr + idx_base + 2, mask=mask2, other=0.0)
        a3 = tl.load(a_ptr + idx_base + 3, mask=mask3, other=0.0)
        a4 = tl.load(a_ptr + idx_base + 4, mask=mask4, other=0.0)
        
        b0 = tl.load(b_ptr + idx_base, mask=mask0, other=0.0)
        b1 = tl.load(b_ptr + idx_base + 1, mask=mask1, other=0.0)
        b2 = tl.load(b_ptr + idx_base + 2, mask=mask2, other=0.0)
        b3 = tl.load(b_ptr + idx_base + 3, mask=mask3, other=0.0)
        b4 = tl.load(b_ptr + idx_base + 4, mask=mask4, other=0.0)
        
        # Compute saxpy operations
        a0 += alpha * b0
        a1 += alpha * b1
        a2 += alpha * b2
        a3 += alpha * b3
        a4 += alpha * b4
        
        # Store results back
        tl.store(a_ptr + idx_base, a0, mask=mask0)
        tl.store(a_ptr + idx_base + 1, a1, mask=mask1)
        tl.store(a_ptr + idx_base + 2, a2, mask=mask2)
        tl.store(a_ptr + idx_base + 3, a3, mask=mask3)
        tl.store(a_ptr + idx_base + 4, a4, mask=mask4)

def s351_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1280  # Multiple of 5 for unrolled access
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s351_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)