import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n_elements, 5 * BLOCK_SIZE):
        current_block_start = i + block_start
        
        for unroll_step in range(5):
            idx_offset = current_block_start + unroll_step * BLOCK_SIZE + offsets
            mask = idx_offset < n_elements
            
            if tl.sum(mask.to(tl.int32)) > 0:
                # Load indices from ip array
                ip_indices = tl.load(ip_ptr + idx_offset, mask=mask, other=0)
                
                # Load current a values
                a_vals = tl.load(a_ptr + idx_offset, mask=mask, other=0.0)
                
                # Gather from b array using ip indices
                b_vals = tl.load(b_ptr + ip_indices, mask=mask, other=0.0)
                
                # Compute saxpy: a[i] += alpha * b[ip[i]]
                result = a_vals + alpha * b_vals
                
                # Store result back
                tl.store(a_ptr + idx_offset, result, mask=mask)

def s353_triton(a, b, c, ip):
    alpha = c[0].item()
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE * 5)
    
    grid = (num_blocks,)
    
    s353_kernel[grid](
        a, b, ip, alpha, n_elements, BLOCK_SIZE
    )