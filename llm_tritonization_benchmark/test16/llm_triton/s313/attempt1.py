import torch
import triton
import triton.language as tl

@triton.jit
def s313_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block dot product
    block_dot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values from arrays a and b
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Multiply and accumulate
        block_dot += a_vals * b_vals
    
    # Sum all elements in the block
    total_dot = tl.sum(block_dot, axis=0)
    
    # Store the result (each block writes its partial sum)
    block_id = tl.program_id(0)
    tl.store(output_ptr + block_id, total_dot)

def s313_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s313_kernel[(n_blocks,)](
        a, b, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_sums).item()
    
    return dot