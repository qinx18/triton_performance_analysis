import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, 
                 len_2d, aa_stride0, aa_stride1, result_ptr,
                 BLOCK_SIZE: tl.constexpr):
    
    # Get program ID for this block
    pid = tl.program_id(0)
    
    # Calculate the range of elements this block will process
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = i_offsets < (len_2d - 1)
    
    # Load indices
    ip_vals = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Calculate offsets for array a
    a_offsets = inc + i_offsets
    
    # Load values from array a
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Calculate 2D indices for aa array
    row_idx = j - 1
    aa_offsets = row_idx * aa_stride0 + ip_vals * aa_stride1
    
    # Load values from aa array
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Compute partial sum for this block
    partial_sum = tl.sum(products)
    
    # Use atomic add to accumulate partial sums
    tl.atomic_add(result_ptr, partial_sum)

def s4116_triton(a, aa, ip, inc, j):
    len_2d = aa.shape[0]
    n_elements = len_2d - 1
    
    # Create a result tensor
    result = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Block size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s4116_kernel[(grid_size,)](
        a,
        aa,
        ip,
        inc,
        j,
        len_2d,
        aa.stride(0),
        aa.stride(1),
        result,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return the sum
    return result.item()