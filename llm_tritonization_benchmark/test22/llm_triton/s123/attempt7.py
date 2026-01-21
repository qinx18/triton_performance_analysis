import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    # Each thread processes one element
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid threads
    mask = tid < n_half
    
    # Load input values
    b_vals = tl.load(b_ptr + tid, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + tid, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + tid, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + tid, mask=mask, other=0.0)
    
    # Calculate base j position for each thread
    # j starts at -1, increments before each store
    # For thread i: base j = i, then potentially i+count_positive_c_before_i
    j_base = tid
    
    # Count positive c values before current position
    j_offset = tl.zeros_like(tid, dtype=tl.int32)
    
    for k in range(n_half):
        # Load c[k] for all threads
        c_k = tl.load(c_ptr + k)
        
        # Add 1 to j_offset for threads where k < tid and c[k] > 0
        condition = (k < tid) & (c_k > 0.0)
        j_offset = tl.where(condition, j_offset + 1, j_offset)
    
    # Calculate actual j positions
    j_pos1 = j_base + j_offset
    j_pos2 = j_pos1 + 1
    
    # First store: a[j] = b[i] + d[i] * e[i]
    val1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_pos1, val1, mask=mask)
    
    # Second conditional store
    c_positive = c_vals > 0.0
    val2 = c_vals + d_vals * e_vals
    store_mask = mask & c_positive
    tl.store(a_ptr + j_pos2, val2, mask=store_mask)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e, n_half, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a