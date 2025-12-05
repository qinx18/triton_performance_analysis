import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n_elements, LEN_2D, 
                 sum_ptr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid elements
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate offset indices for array a
    off_indices = inc + indices
    
    # Load values from array a
    a_vals = tl.load(a_ptr + off_indices, mask=mask, other=0.0)
    
    # Calculate 2D indices for aa array
    row_idx = j - 1
    aa_indices = row_idx * LEN_2D + ip_indices
    
    # Load values from aa array
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Sum the products
    block_sum = tl.sum(products)
    
    # Store partial sum
    tl.atomic_add(sum_ptr, block_sum)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = aa.shape[0] - 1  # LEN_2D - 1
    LEN_2D = aa.shape[0]
    
    # Initialize sum tensor
    sum_tensor = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a, aa, ip, inc, j, n_elements, LEN_2D,
        sum_tensor, BLOCK_SIZE
    )
    
    return sum_tensor.item()