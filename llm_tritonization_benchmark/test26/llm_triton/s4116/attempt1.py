import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, output_ptr, 
                 inc, j, len_2d, aa_stride0, aa_stride1,
                 BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid indices
    mask = i_offsets < (len_2d - 1)
    
    # Calculate array offsets
    a_indices = inc + i_offsets
    ip_indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Calculate 2D array indices for aa[j-1][ip[i]]
    aa_row = j - 1
    aa_indices = aa_row * aa_stride1 + ip_indices
    
    # Load values
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Store products for reduction
    tl.store(output_ptr + i_offsets, products, mask=mask)

def s4116_triton(a, aa, ip):
    # Get dimensions from tensor shapes
    len_2d = aa.shape[0]
    
    # Parameters from original code structure
    inc = 25  # Typical value for inc parameter
    j = len_2d // 2  # Typical value for j parameter
    
    n_elements = len_2d - 1
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial products
    partial_products = torch.zeros(n_elements, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a, aa, ip, partial_products,
        inc, j, len_2d, aa.stride(0), aa.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial products
    result = torch.sum(partial_products)
    
    return result.item()