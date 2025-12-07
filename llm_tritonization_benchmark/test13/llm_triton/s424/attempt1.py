import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array, a, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Get strip ID and calculate strip boundaries
    strip_id = tl.program_id(0)
    strip_start = strip_id * 64
    strip_end = min(strip_start + 64, n_elements)
    
    # Calculate indices for this strip
    idx = strip_start + offsets
    
    # Create masks
    valid_mask = idx < strip_end
    
    # Load values
    flat_vals = tl.load(flat_2d_array + idx, mask=valid_mask, other=0.0)
    a_vals = tl.load(a + idx, mask=valid_mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1] which is flat_2d_array[idx + 64]
    write_idx = idx + 64
    write_mask = valid_mask & (write_idx < (n_elements + 64))
    tl.store(flat_2d_array + write_idx, result, mask=write_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1
    STRIP_SIZE = 64
    BLOCK_SIZE = 64
    
    # Calculate number of strips needed
    num_strips = triton.cdiv(n_elements, STRIP_SIZE)
    
    # Process strips sequentially
    for strip_id in range(num_strips):
        # Launch one thread block per strip
        grid = (1,)
        s424_kernel[grid](
            flat_2d_array, 
            a, 
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Force synchronization between strips
        torch.cuda.synchronize()