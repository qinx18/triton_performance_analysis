import triton
import triton.language as tl
import torch

@triton.jit
def s341_kernel(b_ptr, a_ptr, valid_count_ptr, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # This kernel packs positive values from b into a
    # Since this is inherently sequential, we use a single block approach
    
    block_id = tl.program_id(0)
    if block_id > 0:
        return
    
    # Initialize output counter
    j = 0
    
    # Process elements in chunks
    for start_idx in range(0, LEN_1D, BLOCK_SIZE):
        end_idx = min(start_idx + BLOCK_SIZE, LEN_1D)
        chunk_size = end_idx - start_idx
        
        # Load chunk of b
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        
        # Process each element in the chunk sequentially
        for local_i in range(chunk_size):
            b_val = tl.load(b_ptr + start_idx + local_i)
            if b_val > 0.0:
                tl.store(a_ptr + j, b_val)
                j += 1
    
    # Store the count of valid elements
    tl.store(valid_count_ptr, j)

def s341_triton(a, b):
    LEN_1D = b.shape[0]
    BLOCK_SIZE = 256
    
    # Create a tensor to store the count of valid elements
    valid_count = torch.zeros(1, dtype=torch.int32, device=b.device)
    
    # Launch kernel with single block since operation is inherently sequential
    grid = (1,)
    s341_kernel[grid](
        b, a, valid_count,
        LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a