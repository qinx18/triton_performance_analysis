import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset vectors for vectorized operations
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize dot product accumulator
    dot = 0.0
    
    # Process the array in chunks of 5 elements
    for i in range(0, n_elements, 5):
        # Calculate current offsets for all 5 elements
        offset0 = i + offsets * 5
        offset1 = offset0 + 1
        offset2 = offset0 + 2
        offset3 = offset0 + 3
        offset4 = offset0 + 4
        
        # Create masks for valid elements
        mask0 = offset0 < n_elements
        mask1 = offset1 < n_elements
        mask2 = offset2 < n_elements
        mask3 = offset3 < n_elements
        mask4 = offset4 < n_elements
        
        # Load elements from arrays a and b
        a0 = tl.load(a_ptr + offset0, mask=mask0, other=0.0)
        b0 = tl.load(b_ptr + offset0, mask=mask0, other=0.0)
        
        a1 = tl.load(a_ptr + offset1, mask=mask1, other=0.0)
        b1 = tl.load(b_ptr + offset1, mask=mask1, other=0.0)
        
        a2 = tl.load(a_ptr + offset2, mask=mask2, other=0.0)
        b2 = tl.load(b_ptr + offset2, mask=mask2, other=0.0)
        
        a3 = tl.load(a_ptr + offset3, mask=mask3, other=0.0)
        b3 = tl.load(b_ptr + offset3, mask=mask3, other=0.0)
        
        a4 = tl.load(a_ptr + offset4, mask=mask4, other=0.0)
        b4 = tl.load(b_ptr + offset4, mask=mask4, other=0.0)
        
        # Compute dot products and accumulate
        dot += tl.sum(a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4)
    
    # Store the result
    if pid == 0:
        tl.store(output_ptr, dot)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * 5),)
    
    # Launch kernel
    s352_kernel[grid](
        a, b, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()