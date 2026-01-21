import torch
import triton
import triton.language as tl

@triton.jit
def s251_expand_s_kernel(s_expanded_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Expand scalar s to array using sequential computation"""
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first program handles expansion
        s_val = 0.0
        
        for block_start in range(0, N, BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, N)
            actual_block_size = block_end - block_start
            
            # Load block data
            offsets = tl.arange(0, BLOCK_SIZE)
            block_offsets = block_start + offsets
            mask = offsets < actual_block_size
            
            b_vals = tl.load(b_ptr + block_offsets, mask=mask, other=0.0)
            c_vals = tl.load(c_ptr + block_offsets, mask=mask, other=0.0)
            d_vals = tl.load(d_ptr + block_offsets, mask=mask, other=0.0)
            
            # Process each element in the block sequentially
            for i in range(actual_block_size):
                b_val = tl.load(b_ptr + block_start + i)
                c_val = tl.load(c_ptr + block_start + i)
                d_val = tl.load(d_ptr + block_start + i)
                
                s_val = b_val + c_val * d_val
                tl.store(s_expanded_ptr + block_start + i, s_val)

@triton.jit
def s251_compute_kernel(a_ptr, s_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Compute final values using expanded s array"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] = s * s
    a_vals = s_vals * s_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s251_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded scalar
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s to array
    grid1 = (1,)  # Single program for sequential computation
    s251_expand_s_kernel[grid1](
        s_expanded, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Compute final values in parallel
    grid2 = (triton.cdiv(N, BLOCK_SIZE),)
    s251_compute_kernel[grid2](
        a, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE
    )