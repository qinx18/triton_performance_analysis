import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute base offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process 5 elements at a time within each block
    for local_start in range(0, BLOCK_SIZE, 5):
        # Calculate global indices for this group of 5
        base_idx = block_start + local_start
        
        # Check if we have at least one valid element
        if base_idx >= n_elements:
            break
            
        # Create indices for the 5 elements
        idx_offsets = offsets[local_start:local_start+5] if local_start + 5 <= BLOCK_SIZE else offsets[local_start:]
        current_indices = base_idx + tl.arange(0, 5) if local_start + 5 <= BLOCK_SIZE else base_idx + tl.arange(0, min(5, BLOCK_SIZE - local_start))
        
        # Mask for valid elements
        mask = current_indices < n_elements
        
        # Load indirect indices
        ip_vals = tl.load(ip_ptr + current_indices, mask=mask)
        
        # Load values from b using indirect addressing
        b_vals = tl.load(b_ptr + ip_vals, mask=mask)
        
        # Load current values from a
        a_vals = tl.load(a_ptr + current_indices, mask=mask)
        
        # Perform saxpy operation: a[i] += alpha * b[ip[i]]
        result = a_vals + alpha * b_vals
        
        # Store back to a
        tl.store(a_ptr + current_indices, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0].item()
    
    # Use block size that's divisible by 5 for efficiency
    BLOCK_SIZE = 320  # 64 * 5, good for unrolling by 5
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (grid_size,)
    
    # Launch kernel
    s353_kernel[grid](
        a, b, ip, alpha, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )