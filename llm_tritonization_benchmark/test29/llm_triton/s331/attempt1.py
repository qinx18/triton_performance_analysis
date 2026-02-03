import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find the last index where a[i] < 0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result to -1
    if tl.program_id(0) == 0:
        tl.store(result_ptr, -1)
    
    # Process blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check which elements are < 0
        neg_mask = vals < 0.0
        
        # For each element that's < 0, update the result if it's the latest one
        for i in range(BLOCK_SIZE):
            if mask[i] and neg_mask[i]:
                idx = block_start + i
                # Atomically update with max to ensure we get the last occurrence
                tl.atomic_max(result_ptr, idx)

def s331_triton(a):
    N = a.shape[0]
    
    # Output tensor for result
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    s331_kernel[(num_blocks,)](
        a,
        result,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()