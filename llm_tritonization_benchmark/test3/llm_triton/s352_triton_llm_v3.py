import triton
import triton.language as tl
import torch

@triton.jit
def s352_kernel(a_ptr, b_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the range of elements this program will process
    block_start = pid * BLOCK_SIZE
    
    # Initialize accumulator
    dot_acc = 0.0
    
    # Process elements in chunks of 5 (unrolled loop)
    for base_idx in range(block_start, min(block_start + BLOCK_SIZE, n_elements), 5):
        # Create offsets for the 5 elements
        offsets = base_idx + tl.arange(0, 5)
        
        # Create mask to handle boundary conditions
        mask = offsets < n_elements
        
        # Load elements from a and b arrays
        a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        
        # Compute dot product for this group of 5 elements
        products = a_vals * b_vals
        dot_acc += tl.sum(products)
    
    # Store the partial result
    tl.store(result_ptr + pid, dot_acc)

def s352_triton(a, b):
    n_elements = a.shape[0]
    
    # Choose block size to process elements in groups of 5
    BLOCK_SIZE = 320  # Multiple of 5 for efficient unrolling
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial results
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s352_kernel[(num_blocks,)](
        a, b, partial_results,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final dot product
    dot = torch.sum(partial_results)
    
    return dot