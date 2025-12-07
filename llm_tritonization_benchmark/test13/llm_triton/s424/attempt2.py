import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array, a, strip_start, strip_size, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Masks
    valid_mask = (idx >= strip_start) & (idx < (strip_start + strip_size)) & (idx < n_elements)
    write_idx = idx + 64
    
    # Load values
    flat_vals = tl.load(flat_2d_array + idx, mask=valid_mask, other=0.0)
    a_vals = tl.load(a + idx, mask=valid_mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store result
    tl.store(flat_2d_array + write_idx, result, mask=valid_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Process strips sequentially
    for strip_start in range(0, n_elements, STRIP_SIZE):
        strip_size = min(STRIP_SIZE, n_elements - strip_start)
        
        # Launch kernel for this strip
        grid = (triton.cdiv(strip_size, BLOCK_SIZE),)
        s424_kernel[grid](
            flat_2d_array, 
            a, 
            strip_start,
            strip_size,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Synchronize to ensure strip completes before next one
        torch.cuda.synchronize()