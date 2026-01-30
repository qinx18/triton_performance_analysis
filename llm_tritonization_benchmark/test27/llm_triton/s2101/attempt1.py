import triton
import triton.language as tl
import torch

@triton.jit
def s2101_kernel(aa_ptr, bb_ptr, cc_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load diagonal elements
    aa_vals = tl.load(aa_ptr + offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + offsets, mask=mask)
    
    # Compute aa[i][i] += bb[i][i] * cc[i][i]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(aa_ptr + offsets, result, mask=mask)

def s2101_triton(aa, bb, cc, len_2d):
    n_elements = len_2d
    
    # Create offset arrays for diagonal access
    diag_indices = torch.arange(n_elements, device=aa.device, dtype=torch.long)
    flat_indices = diag_indices * len_2d + diag_indices
    
    # Get diagonal elements as contiguous arrays
    aa_diag = aa.view(-1)[flat_indices]
    bb_diag = bb.view(-1)[flat_indices] 
    cc_diag = cc.view(-1)[flat_indices]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2101_kernel[grid](
        aa_diag, bb_diag, cc_diag,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Write back diagonal elements to original array
    aa.view(-1)[flat_indices] = aa_diag