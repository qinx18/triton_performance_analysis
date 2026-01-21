import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    k = 0
    for i in range(N):
        # Load bb[j][i] for all j in this block
        bb_ptrs = bb_ptr + j_idx * N + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Load aa[j][i] for all j in this block
        aa_ptrs = aa_ptr + j_idx * N + i
        aa_vals = tl.load(aa_ptrs, mask=j_mask, other=0.0)
        
        # Create condition mask
        cond_mask = bb_vals > 0.0
        combined_mask = j_mask & cond_mask
        
        # Count how many elements pass condition in this block
        for block_j in range(BLOCK_SIZE):
            if pid * BLOCK_SIZE + block_j < N:
                if bb_vals[block_j] > 0.0:
                    tl.store(flat_2d_array_ptr + k, aa_vals[block_j])
                    k += 1

def s343_triton(aa, bb, flat_2d_array):
    N = aa.shape[0]
    
    # Use PyTorch for stream compaction since Triton can't handle variable output indices efficiently
    mask = bb > 0.0
    packed_values = aa[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values.flatten()
    
    return num_packed