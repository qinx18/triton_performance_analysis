import triton
import triton.language as tl
import torch

@triton.jit
def s317_kernel(output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        # Compute q = 0.99^(n//2)
        num_iterations = n // 2
        
        # Use logarithmic reduction for better parallelization
        q = 1.0
        factor = 0.99
        
        # Process powers of 2
        power = 1
        current_factor = factor
        
        while power <= num_iterations:
            if (num_iterations & power) != 0:
                q = q * current_factor
            current_factor = current_factor * current_factor
            power = power * 2
        
        tl.store(output_ptr, q)

@triton.jit  
def s317_parallel_kernel(output_ptr, log_values_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    num_iterations = n // 2
    mask = offsets < num_iterations
    
    # Each thread computes log(0.99) for its assigned iterations
    log_factor = tl.log(0.99)
    log_values = tl.where(mask, log_factor, 0.0)
    
    # Store partial results
    tl.store(log_values_ptr + offsets, log_values, mask=mask)

def s317_triton(n):
    num_iterations = n // 2
    
    if num_iterations == 0:
        return 1.0
    
    # For small problems, use simple kernel
    if num_iterations < 1024:
        output = torch.zeros(1, dtype=torch.float32, device='cuda')
        BLOCK_SIZE = 256
        grid = (1,)
        s317_kernel[grid](output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output.item()
    
    # For large problems, use parallel reduction
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    # Allocate temporary storage for log values
    log_values = torch.zeros(num_iterations, dtype=torch.float32, device='cuda')
    
    # Compute log(0.99) for each iteration in parallel
    s317_parallel_kernel[grid](
        torch.zeros(1, dtype=torch.float32, device='cuda'),
        log_values,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all log values and exponentiate
    total_log = torch.sum(log_values)
    result = torch.exp(total_log)
    
    return result.item()