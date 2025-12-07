import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Sequential wavefront pattern - process anti-diagonals
    for diag in range(2, 2 * N - 1):  # diag = i + j
        start_i = tl.maximum(1, diag - N + 1)
        end_i = tl.minimum(diag, N)
        
        # Get program ID for this diagonal
        pid = tl.program_id(0)
        
        # Calculate number of elements on this diagonal
        diag_size = end_i - start_i
        
        # Block processing for elements on this diagonal
        block_start = pid * BLOCK_SIZE
        if block_start >= diag_size:
            continue
            
        # Generate offsets for this block
        offsets = tl.arange(0, BLOCK_SIZE)
        element_ids = block_start + offsets
        mask = element_ids < diag_size
        
        # Calculate actual i, j coordinates
        i_coords = start_i + element_ids
        j_coords = diag - i_coords
        
        # Bounds check
        valid_mask = mask & (i_coords >= 1) & (i_coords < N) & (j_coords >= 1) & (j_coords < N)
        
        # Calculate memory offsets
        aa_offsets = i_coords * N + j_coords
        aa_dep_offsets = (i_coords - 1) * N + (j_coords - 1)
        bb_offsets = i_coords * N + j_coords
        
        # Load values
        aa_dep_vals = tl.load(aa_ptr + aa_dep_offsets, mask=valid_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=valid_mask, other=0.0)
        
        # Compute result
        result = aa_dep_vals + bb_vals
        
        # Store result
        tl.store(aa_ptr + aa_offsets, result, mask=valid_mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Sequential processing with limited parallelism per diagonal
    for diag in range(2, 2 * N - 1):
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        diag_size = end_i - start_i
        
        if diag_size <= 0:
            continue
            
        grid = (triton.cdiv(diag_size, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, N, BLOCK_SIZE
        )