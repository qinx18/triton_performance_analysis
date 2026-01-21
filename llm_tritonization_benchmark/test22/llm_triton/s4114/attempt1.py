import triton
import triton.language as tl

@triton.jit
def s4114_kernel(a_ptr, b_ptr, c_ptr, d_ptr, ip_ptr, n1, n, len_c, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate indices: i ranges from n1-1 to n-1
    i_indices = block_start + offsets
    
    # Mask for valid indices in range [n1-1, n)
    mask = (i_indices >= n1 - 1) & (i_indices < n)
    
    # Load ip[i] with masking
    ip_vals = tl.load(ip_ptr + i_indices, mask=mask)
    
    # Calculate c array index: LEN_1D - k + 1 - 2 = n - ip[i] - 1
    c_indices = n - ip_vals - 1
    
    # Load arrays with masking
    b_vals = tl.load(b_ptr + i_indices, mask=mask)
    d_vals = tl.load(d_ptr + i_indices, mask=mask)
    
    # Mask for valid c indices
    c_mask = mask & (c_indices >= 0) & (c_indices < len_c)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute: a[i] = b[i] + c[n-k-1] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + i_indices, result, mask=mask)

def s4114_triton(a, b, c, d, ip, n1):
    n = a.shape[0]
    len_c = c.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s4114_kernel[grid](
        a, b, c, d, ip,
        n1, n, len_c,
        BLOCK_SIZE=BLOCK_SIZE
    )