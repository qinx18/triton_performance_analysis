import triton
import triton.language as tl

@triton.jit
def s4112_kernel(a_ptr, b_ptr, ip_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Load values from b array using gathered indices
    b_vals = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Load current values from a array
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store results back to a array
    tl.store(a_ptr + indices, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )