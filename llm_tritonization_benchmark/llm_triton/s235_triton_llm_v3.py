import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(
    a_ptr, b_ptr, c_ptr, aa_ptr, aa_copy_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Load 1D arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Update a[i] += b[i] * c[i]
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Inner loop: for j = 1 to LEN_2D-1
    for j in range(1, LEN_2D):
        j_prev = j - 1
        
        # Load aa[j-1][i] from copy (read-only)
        aa_prev_offsets = j_prev * LEN_2D + offsets
        aa_prev_vals = tl.load(aa_copy_ptr + aa_prev_offsets, mask=mask)
        
        # Load bb[j][i]
        bb_offsets = j * LEN_2D + offsets
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_vals = aa_prev_vals + bb_vals * a_vals
        
        # Store to original aa array
        aa_write_offsets = j * LEN_2D + offsets
        tl.store(aa_ptr + aa_write_offsets, aa_vals, mask=mask)

def s235_triton(a, b, c, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy for WAR dependency handling
    aa_copy = aa.clone()
    
    # Calculate grid size
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Launch kernel
    s235_kernel[(grid_size,)](
        a, b, c, aa, aa_copy, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )