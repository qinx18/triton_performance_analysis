import triton
import triton.language as tl

@triton.jit
def s4112_kernel(a_ptr, b_ptr, ip_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices for indirect addressing
    ip_vals = tl.load(ip_ptr + indices, mask=mask)
    
    # Load values from b using indirect addressing (gather)
    b_vals = tl.load(b_ptr + ip_vals, mask=mask)
    
    # Load current values from a
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Perform saxpy operation: a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store result back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )