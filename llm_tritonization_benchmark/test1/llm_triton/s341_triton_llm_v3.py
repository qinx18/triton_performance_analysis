import triton
import triton.language as tl
import torch

@triton.jit
def s341_kernel(b_ptr, a_ptr, output_count_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel packs positive values from b into a
    # Each block processes a chunk of the input array
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from b
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Find positive values
    pos_mask = (b_vals > 0.0) & mask
    
    # Count positive values in this block
    local_count = tl.sum(pos_mask.to(tl.int32))
    
    # Get global offset for this block's positive values
    block_offset = tl.atomic_add(output_count_ptr, local_count)
    
    # Compact and store positive values
    write_offset = 0
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            val = tl.load(b_ptr + block_start + i)
            if val > 0.0:
                tl.store(a_ptr + block_offset + write_offset, val)
                write_offset += 1

def s341_triton(a, b):
    n_elements = b.numel()
    
    # Initialize output count
    output_count = torch.zeros(1, dtype=torch.int32, device=b.device)
    
    # Clear the output array
    a.zero_()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s341_kernel[grid](
        b, a, output_count,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a