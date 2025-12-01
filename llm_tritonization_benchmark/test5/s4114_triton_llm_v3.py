import torch
import triton
import triton.language as tl

@triton.jit
def s4114_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr,
    n1, len_1d,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(n1 - 1, len_1d, BLOCK_SIZE):
        i_offsets = block_start + offsets
        mask = i_offsets < len_1d
        
        # Load ip[i] values
        k_vals = tl.load(ip_ptr + i_offsets, mask=mask)
        
        # Load b[i] and d[i]
        b_vals = tl.load(b_ptr + i_offsets, mask=mask)
        d_vals = tl.load(d_ptr + i_offsets, mask=mask)
        
        # Calculate c indices: LEN_1D - k + 1 - 2 = LEN_1D - k - 1
        c_indices = len_1d - k_vals - 1
        
        # Load c values using gathered indices
        c_vals = tl.load(c_ptr + c_indices, mask=mask)
        
        # Compute a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
        result = b_vals + c_vals * d_vals
        
        # Store result
        tl.store(a_ptr + i_offsets, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    len_1d = a.shape[0]
    
    BLOCK_SIZE = 256
    
    s4114_kernel[(1,)](
        a, b, c, d, ip,
        n1, len_1d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a