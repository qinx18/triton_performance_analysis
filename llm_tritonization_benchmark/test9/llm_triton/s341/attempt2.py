import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(a_ptr, b_ptr, output_count_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel packs positive values from b into a
    # Since this is inherently sequential, we need to process sequentially
    
    # Initialize output count to 0
    if tl.program_id(0) == 0:
        tl.store(output_count_ptr, 0)
    
    # Use a single thread to process sequentially
    if tl.program_id(0) != 0:
        return
        
    offsets = tl.arange(0, BLOCK_SIZE)
    j = 0
    
    # Process in blocks but maintain sequential order
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of b values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        i = 0
        while i < BLOCK_SIZE:
            element_idx = block_start + i
            if element_idx >= n_elements:
                i = BLOCK_SIZE  # Exit loop
            else:
                # Get the value for current element
                b_val = tl.load(b_ptr + element_idx)
                
                if b_val > 0.0:
                    tl.store(a_ptr + j, b_val)
                    j += 1
                i += 1
    
    # Store final count
    tl.store(output_count_ptr, j)

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output count tensor
    output_count = torch.zeros(1, dtype=torch.int32, device=b.device)
    
    # Launch kernel with single block since this is inherently sequential
    grid = (1,)
    
    s341_kernel[grid](
        a, b, output_count,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, output_count