import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    for t in range(TMAX):
        # First loop: for (j = 0; j < NY; j++) ey[0][j] = _fict_[t];
        fict_val = tl.load(_fict_ptr + t)
        j_offsets = tl.arange(0, BLOCK_SIZE)
        for j_block in range(0, NY, BLOCK_SIZE):
            j_idx = j_block + j_offsets
            j_mask = j_idx < NY
            ey_idx = 0 * NY + j_idx  # ey[0][j]
            tl.store(ey_ptr + ey_idx, fict_val, mask=j_mask)
        
        # Second loop: for (i = 1; i < NX; i++) for (j = 0; j < NY; j++) ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
        ij_offsets = tl.arange(0, BLOCK_SIZE)
        for ij_block in range(0, (NX-1) * NY, BLOCK_SIZE):
            ij_idx = ij_block + ij_offsets
            ij_mask = ij_idx < (NX-1) * NY
            
            i = (ij_idx // NY) + 1  # i starts from 1
            j = ij_idx % NY
            
            ey_idx = i * NY + j
            hz_idx = i * NY + j
            hz_prev_idx = (i-1) * NY + j
            
            ey_val = tl.load(ey_ptr + ey_idx, mask=ij_mask)
            hz_val = tl.load(hz_ptr + hz_idx, mask=ij_mask)
            hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=ij_mask)
            
            new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
            tl.store(ey_ptr + ey_idx, new_ey, mask=ij_mask)
        
        # Third loop: for (i = 0; i < NX; i++) for (j = 1; j < NY; j++) ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
        ij_offsets = tl.arange(0, BLOCK_SIZE)
        for ij_block in range(0, NX * (NY-1), BLOCK_SIZE):
            ij_idx = ij_block + ij_offsets
            ij_mask = ij_idx < NX * (NY-1)
            
            i = ij_idx // (NY-1)
            j = (ij_idx % (NY-1)) + 1  # j starts from 1
            
            ex_idx = i * NY + j
            hz_idx = i * NY + j
            hz_prev_idx = i * NY + (j-1)
            
            ex_val = tl.load(ex_ptr + ex_idx, mask=ij_mask)
            hz_val = tl.load(hz_ptr + hz_idx, mask=ij_mask)
            hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=ij_mask)
            
            new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
            tl.store(ex_ptr + ex_idx, new_ex, mask=ij_mask)
        
        # Fourth loop: for (i = 0; i < NX - 1; i++) for (j = 0; j < NY - 1; j++) hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
        ij_offsets = tl.arange(0, BLOCK_SIZE)
        for ij_block in range(0, (NX-1) * (NY-1), BLOCK_SIZE):
            ij_idx = ij_block + ij_offsets
            ij_mask = ij_idx < (NX-1) * (NY-1)
            
            i = ij_idx // (NY-1)
            j = ij_idx % (NY-1)
            
            hz_idx = i * NY + j
            ex_idx = i * NY + j
            ex_next_idx = i * NY + (j+1)
            ey_idx = i * NY + j
            ey_next_idx = (i+1) * NY + j
            
            hz_val = tl.load(hz_ptr + hz_idx, mask=ij_mask)
            ex_val = tl.load(ex_ptr + ex_idx, mask=ij_mask)
            ex_next_val = tl.load(ex_ptr + ex_next_idx, mask=ij_mask)
            ey_val = tl.load(ey_ptr + ey_idx, mask=ij_mask)
            ey_next_val = tl.load(ey_ptr + ey_next_idx, mask=ij_mask)
            
            new_hz = hz_val - 0.7 * (ex_next_val - ex_val + ey_next_val - ey_val)
            tl.store(hz_ptr + hz_idx, new_hz, mask=ij_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 256
    grid = (1,)
    
    fdtd_2d_kernel[grid](
        _fict_, ex, ey, hz,
        NX, NY, TMAX,
        BLOCK_SIZE
    )