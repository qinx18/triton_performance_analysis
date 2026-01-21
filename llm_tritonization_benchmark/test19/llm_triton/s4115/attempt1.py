import torch
import triton
import triton.language as tl

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start offset
    block_start = pid * BLOCK_SIZE
    
    # Create offset vectors
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    local_sum = 0.0
    
    # Process elements in this block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n:
            # Load a[idx]
            a_val = tl.load(a_ptr + idx)
            
            # Load ip[idx] and use it to index into b
            ip_val = tl.load(ip_ptr + idx)
            b_val = tl.load(b_ptr + ip_val)
            
            # Accumulate dot product
            local_sum += a_val * b_val
    
    # Store partial sum
    tl.store(output_ptr + pid, local_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    
    # Calculate grid size
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4115_kernel[grid](a, b, ip, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all partial results
    result = torch.sum(partial_sums)
    
    return result.item()