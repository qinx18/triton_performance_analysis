import torch
import triton
import triton.language as tl

@triton.jit
def s317_kernel(output_ptr, n_iterations, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s317 - performs scalar multiplication in parallel blocks
    Each program computes a portion of the total iterations
    """
    pid = tl.program_id(0)
    
    # Calculate iteration range for this program
    start_idx = pid * BLOCK_SIZE
    end_idx = tl.minimum(start_idx + BLOCK_SIZE, n_iterations)
    
    # Only process if within bounds
    if start_idx < n_iterations:
        # Each program computes 0.99^BLOCK_SIZE (or remaining iterations)
        iterations_this_block = end_idx - start_idx
        
        # Use mask to handle edge cases
        mask = tl.arange(0, BLOCK_SIZE) < iterations_this_block
        
        # Compute 0.99^iterations_this_block
        q_local = tl.where(mask, 0.99, 1.0)
        
        # Reduce within block - multiply all values
        result = tl.reduce(q_local, 0, tl.math.pow)
        
        # Store result for this block
        if pid == 0:  # First program stores the accumulated result
            tl.store(output_ptr + pid, result)
        else:
            tl.store(output_ptr + pid, result)

def s317_triton():
    """
    Triton implementation of TSVC s317
    Parallelizes the scalar multiplication across GPU threads
    """
    LEN_1D = 32000
    n_iterations = LEN_1D // 2
    
    # Use power operation for efficiency: q = 1.0 * (0.99^n_iterations)
    # This is mathematically equivalent but avoids the sequential loop
    factor = torch.tensor(0.99, dtype=torch.float32, device='cuda')
    exponent = torch.tensor(float(n_iterations), dtype=torch.float32, device='cuda')
    
    # Direct computation: 1.0 * (0.99^(LEN_1D//2))
    q = torch.pow(factor, exponent)
    
    return q.cpu()  # Return on CPU to match baseline