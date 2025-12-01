import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the range of elements this program will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load indices for indirect addressing
    ip_vals = tl.load(ip_ptr + offsets, mask=mask, other=0)
    
    # Load b values using indirect addressing
    b_vals = tl.load(b_ptr + ip_vals, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * b_vals
    
    # Sum the products
    result = tl.sum(products)
    
    # Store the partial sum
    result_ptr = tl.program_id(axis=0)
    tl.store(tl.pointer_type(tl.float32)(result_ptr), result)

@triton.jit
def s4115_reduce_kernel(partial_sums_ptr, n_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:  # Only first program does the reduction
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_blocks
        
        partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
        total_sum = tl.sum(partial_sums)
        
        # Store final result at index 0
        tl.store(partial_sums_ptr, total_sum)

def s4115_triton(a, b, ip):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate memory for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=torch.float32, device=a.device)
    
    # Launch the main kernel
    grid = (n_blocks,)
    s4115_kernel[grid](
        a, b, ip, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # If we have multiple blocks, we need to reduce the partial sums
    if n_blocks > 1:
        reduce_grid = (1,)
        reduce_block_size = min(1024, triton.next_power_of_2(n_blocks))
        s4115_reduce_kernel[reduce_grid](
            partial_sums, n_blocks, BLOCK_SIZE=reduce_block_size
        )
    
    return partial_sums[0].item()