import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, result_ptr, 
                 inc, j, len_2d, aa_stride0, aa_stride1,
                 BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < (len_2d - 1)
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute a offsets: off = inc + i
    a_offsets = inc + indices
    
    # Load from a array
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Compute aa offsets: aa[j-1][ip[i]]
    row_idx = j - 1
    aa_offsets = row_idx * aa_stride0 + ip_indices * aa_stride1
    
    # Load from aa array
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Store results for reduction
    tl.store(result_ptr + indices, products, mask=mask)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    # Create output tensor for partial results
    temp_result = torch.zeros(len_2d - 1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d - 1, BLOCK_SIZE),)
    
    # Get strides for 2D array
    aa_stride0 = aa.stride(0)
    aa_stride1 = aa.stride(1)
    
    s4116_kernel[grid](
        a, aa, ip, temp_result,
        inc, j, len_2d, aa_stride0, aa_stride1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum the partial results
    return temp_result.sum().item()