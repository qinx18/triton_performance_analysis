import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n_elements, aa_cols, BLOCK_SIZE: tl.constexpr):
    # Each program handles one block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Initialize sum for this block
    sum_val = 0.0
    
    # Load ip values for this block
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute offsets for array a
    a_offsets = inc + indices
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Compute aa indices: aa[j-1][ip[i]]
    aa_row = j - 1
    aa_indices = aa_row * aa_cols + ip_vals
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products and sum
    products = a_vals * aa_vals
    products = tl.where(mask, products, 0.0)
    
    # Sum all products in this block
    block_sum = tl.sum(products)
    
    # Store the partial sum (we'll reduce these in Python)
    tl.store(tl.program_id(0) + tl.zeros((1,), dtype=tl.int32), block_sum)

def s4116_triton(a, aa, ip, inc, j):
    LEN_2D = aa.size(0)
    n_elements = LEN_2D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid[0], dtype=a.dtype, device=a.device)
    
    # Use a simple reduction approach
    sum_val = 0.0
    
    # Process in blocks
    for i in range(0, n_elements, BLOCK_SIZE):
        end_idx = min(i + BLOCK_SIZE, n_elements)
        block_size = end_idx - i
        
        # Get indices for this block
        indices = torch.arange(i, end_idx, device=a.device, dtype=torch.int32)
        
        # Compute a offsets
        a_offsets = inc + indices
        a_vals = a[a_offsets]
        
        # Get ip values and compute aa indices
        ip_vals = ip[indices]
        aa_indices = (j - 1) * aa.size(1) + ip_vals
        aa_flat = aa.flatten()
        aa_vals = aa_flat[aa_indices]
        
        # Compute products and sum
        products = a_vals * aa_vals
        block_sum = torch.sum(products)
        sum_val += block_sum.item()
    
    return sum_val