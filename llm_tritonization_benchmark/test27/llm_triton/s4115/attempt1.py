import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load ip[i] (indices)
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] using gathered indices
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Compute products and accumulate
        products = a_vals * b_vals
        accumulator += products
    
    # Reduce the accumulator to a scalar
    total_sum = tl.sum(accumulator)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, total_sum)

def s4115_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor
    result = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s4115_kernel[grid](
        a, b, ip, result,
        N, BLOCK_SIZE
    )
    
    return result.item()