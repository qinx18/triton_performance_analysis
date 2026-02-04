import triton
import triton.language as tl
import torch

@triton.jit
def s332_kernel(a_ptr, output_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize return values
    found_value = -1.0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any value in block is greater than threshold
        greater_mask = a_vals > t
        
        # If we find values greater than threshold, take the first one
        if tl.any(greater_mask):
            # Find first valid element in this block
            for i in range(BLOCK_SIZE):
                element_offset = block_start + i
                if element_offset < n_elements:
                    element_mask = element_offset < n_elements
                    if element_mask:
                        val = tl.load(a_ptr + element_offset)
                        if val > t:
                            found_value = val
                            tl.store(output_ptr, found_value)
                            return
    
    # If no value found, store -1.0
    tl.store(output_ptr, found_value)

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Use PyTorch for this search operation since Triton struggles with early termination
    mask = a > t
    indices = torch.nonzero(mask, as_tuple=True)[0]
    
    if len(indices) > 0:
        # Found at least one value greater than threshold
        first_idx = indices[0]
        value = a[first_idx].item()
    else:
        # No value found
        value = -1.0
    
    return value