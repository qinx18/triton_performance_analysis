import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize search variables
    found_index = -2
    found_value = -1.0
    found = False
    
    # Search through array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current block
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Check if any values in this block are greater than threshold
        greater_mask = vals > t
        has_match = tl.sum(greater_mask.to(tl.int32)) > 0
        
        if has_match and not found:
            # Find first matching element in this block
            for i in range(BLOCK_SIZE):
                if i + block_start < n_elements:
                    val = tl.load(a_ptr + (i + block_start))
                    if val > t:
                        found_index = i + block_start
                        found_value = val
                        found = True
                        break
            if found:
                break
    
    # Calculate checksum
    chksum = found_value + found_index
    
    # Store results
    tl.store(output_ptr, found_value)
    tl.store(output_ptr + 1, found_index)
    tl.store(output_ptr + 2, chksum)

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Convert t to scalar if it's a tensor
    if torch.is_tensor(t):
        t = t.item()
    
    # Output tensor to store [value, index, chksum]
    output = torch.zeros(3, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s332_kernel[grid](
        a, t, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output[0].item()  # Return the found value