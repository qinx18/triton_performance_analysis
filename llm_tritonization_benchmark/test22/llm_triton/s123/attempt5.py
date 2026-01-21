import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    start_i = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = start_i + offsets
    
    mask = i_offsets < n_half
    
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # The j index starts from each i and increments conditionally
    # We need to compute the cumulative sum of conditions up to each point
    condition_met = c_vals > 0.0
    
    # For each i, j starts at 2*i (since j increments once per iteration plus conditionally)
    # We need to handle the irregular spacing due to conditional increments
    
    # First store: a[j] = b[i] + d[i] * e[i] where j follows the pattern
    val1 = b_vals + d_vals * e_vals
    
    # For sequential processing of irregular j pattern
    for block_start in range(0, n_half, BLOCK_SIZE):
        if block_start != start_i:
            continue
            
        for local_i in range(BLOCK_SIZE):
            global_i = block_start + local_i
            if global_i >= n_half:
                break
                
            # Calculate actual j position by counting previous conditions
            j_pos = global_i
            for prev_i in range(global_i):
                if prev_i < n_half:
                    j_pos += 1  # Base increment
                    prev_c = tl.load(c_ptr + prev_i)
                    if prev_c > 0.0:
                        j_pos += 1  # Conditional increment
            
            # Load current values
            b_val = tl.load(b_ptr + global_i)
            c_val = tl.load(c_ptr + global_i) 
            d_val = tl.load(d_ptr + global_i)
            e_val = tl.load(e_ptr + global_i)
            
            # First assignment: a[j] = b[i] + d[i] * e[i]
            tl.store(a_ptr + j_pos, b_val + d_val * e_val)
            
            # Conditional assignment
            if c_val > 0.0:
                j_pos += 1
                tl.store(a_ptr + j_pos, c_val + d_val * e_val)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    BLOCK_SIZE = 1  # Use small blocks due to sequential dependency
    
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e, n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )