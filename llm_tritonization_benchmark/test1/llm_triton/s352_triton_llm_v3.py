import torch
import triton
import triton.language as tl

@triton.jit
def s352_kernel(a_ptr, b_ptr, dot_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset arrays for the 5-element unrolled pattern
    offsets_0 = block_start + tl.arange(0, BLOCK_SIZE) * 5
    offsets_1 = offsets_0 + 1
    offsets_2 = offsets_0 + 2
    offsets_3 = offsets_0 + 3
    offsets_4 = offsets_0 + 4
    
    # Create masks for boundary checking
    mask_0 = offsets_0 < n_elements
    mask_1 = offsets_1 < n_elements
    mask_2 = offsets_2 < n_elements
    mask_3 = offsets_3 < n_elements
    mask_4 = offsets_4 < n_elements
    
    # Load values from arrays a and b with masking
    a_0 = tl.load(a_ptr + offsets_0, mask=mask_0, other=0.0)
    b_0 = tl.load(b_ptr + offsets_0, mask=mask_0, other=0.0)
    
    a_1 = tl.load(a_ptr + offsets_1, mask=mask_1, other=0.0)
    b_1 = tl.load(b_ptr + offsets_1, mask=mask_1, other=0.0)
    
    a_2 = tl.load(a_ptr + offsets_2, mask=mask_2, other=0.0)
    b_2 = tl.load(b_ptr + offsets_2, mask=mask_2, other=0.0)
    
    a_3 = tl.load(a_ptr + offsets_3, mask=mask_3, other=0.0)
    b_3 = tl.load(b_ptr + offsets_3, mask=mask_3, other=0.0)
    
    a_4 = tl.load(a_ptr + offsets_4, mask=mask_4, other=0.0)
    b_4 = tl.load(b_ptr + offsets_4, mask=mask_4, other=0.0)
    
    # Compute products and sum them up
    products = a_0 * b_0 + a_1 * b_1 + a_2 * b_2 + a_3 * b_3 + a_4 * b_4
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Store the block sum
    tl.store(dot_ptr + pid, block_sum)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Calculate number of blocks needed (each block processes BLOCK_SIZE * 5 elements)
    BLOCK_SIZE = 256
    elements_per_block = BLOCK_SIZE * 5
    num_blocks = triton.cdiv(n_elements, elements_per_block)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = (num_blocks,)
    s352_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        dot_ptr=partial_sums,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_sums)
    
    return dot