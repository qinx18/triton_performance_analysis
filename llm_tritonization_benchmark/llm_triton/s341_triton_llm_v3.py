import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, valid_count_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel packs positive values from b into a contiguous array a
    # Since this involves dynamic indexing, we need to use atomic operations
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from b
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Check which values are positive
    positive_mask = b_vals > 0.0
    
    # For each positive value, atomically increment counter and store
    for i in range(BLOCK_SIZE):
        if mask[i] and positive_mask[i]:
            # Atomically get next available index
            idx = tl.atomic_add(valid_count_ptr, 1)
            # Store the positive value at that index
            tl.store(a_ptr + idx, b_vals[i])

def s341_triton(a, b):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Counter for valid elements
    valid_count = torch.zeros(1, dtype=torch.int32, device=b.device)
    
    # Clear the output array
    a.zero_()
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s341_kernel[grid](
        b, a, valid_count,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a