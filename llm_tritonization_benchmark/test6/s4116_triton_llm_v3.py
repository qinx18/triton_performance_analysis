import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j_idx, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum accumulator
    sum_val = 0.0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load indices from ip array
        ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Calculate offset for array a
        a_offsets = inc + current_offsets
        a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        
        # Calculate aa indices (j_idx * 256 + ip_vals for flattened 2D array)
        aa_indices = j_idx * 256 + ip_vals
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
        
        # Multiply and accumulate
        products = a_vals * aa_vals
        sum_val += tl.sum(products)
    
    # Store result
    tl.store(tl.program_id(0) + tl.zeros((1,), dtype=tl.float32), sum_val)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = 256 - 1  # LEN_2D - 1
    BLOCK_SIZE = 64
    
    # Flatten aa array if it's 2D
    if aa.dim() == 2:
        aa_flat = aa.flatten()
    else:
        aa_flat = aa
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    grid = (1,)
    s4116_kernel[grid](
        a, aa_flat, ip, 
        inc, j-1,  # j-1 for 0-based indexing
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()