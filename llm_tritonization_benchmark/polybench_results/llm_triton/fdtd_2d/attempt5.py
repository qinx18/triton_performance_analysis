import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    ex_ptr, ey_ptr, hz_ptr, _fict_ptr,
    NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_I)
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    
    for t in range(TMAX):
        # Phase 1: ey[0][j] = _fict_[t]
        if pid_i == 0:
            j_mask = j_offsets < NY
            fict_val = tl.load(_fict_ptr + t)
            ey_offsets = j_offsets
            tl.store(ey_ptr + ey_offsets, fict_val, mask=j_mask)
        
        # Phase 2: ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j])
        i_mask = (i_offsets >= 1) & (i_offsets < NX)
        j_mask = j_offsets < NY
        mask = i_mask[:, None] & j_mask[None, :]
        
        ey_offsets = i_offsets[:, None] * NY + j_offsets[None, :]
        hz_offsets = i_offsets[:, None] * NY + j_offsets[None, :]
        hz_prev_offsets = (i_offsets[:, None] - 1) * NY + j_offsets[None, :]
        
        ey_vals = tl.load(ey_ptr + ey_offsets, mask=mask)
        hz_vals = tl.load(hz_ptr + hz_offsets, mask=mask)
        hz_prev_vals = tl.load(hz_ptr + hz_prev_offsets, mask=mask)
        
        new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
        tl.store(ey_ptr + ey_offsets, new_ey, mask=mask)
        
        # Phase 3: ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1])
        i_mask = i_offsets < NX
        j_mask = (j_offsets >= 1) & (j_offsets < NY)
        mask = i_mask[:, None] & j_mask[None, :]
        
        ex_offsets = i_offsets[:, None] * NY + j_offsets[None, :]
        hz_offsets = i_offsets[:, None] * NY + j_offsets[None, :]
        hz_prev_offsets = i_offsets[:, None] * NY + (j_offsets[None, :] - 1)
        
        ex_vals = tl.load(ex_ptr + ex_offsets, mask=mask)
        hz_vals = tl.load(hz_ptr + hz_offsets, mask=mask)
        hz_prev_vals = tl.load(hz_ptr + hz_prev_offsets, mask=mask)
        
        new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
        tl.store(ex_ptr + ex_offsets, new_ex, mask=mask)
        
        # Phase 4: hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
        i_mask = i_offsets < (NX - 1)
        j_mask = j_offsets < (NY - 1)
        mask = i_mask[:, None] & j_mask[None, :]
        
        hz_offsets = i_offsets[:, None] * NY + j_offsets[None, :]
        ex_curr_offsets = i_offsets[:, None] * NY + j_offsets[None, :]
        ex_next_offsets = i_offsets[:, None] * NY + (j_offsets[None, :] + 1)
        ey_curr_offsets = i_offsets[:, None] * NY + j_offsets[None, :]
        ey_next_offsets = (i_offsets[:, None] + 1) * NY + j_offsets[None, :]
        
        hz_vals = tl.load(hz_ptr + hz_offsets, mask=mask)
        ex_curr_vals = tl.load(ex_ptr + ex_curr_offsets, mask=mask)
        ex_next_vals = tl.load(ex_ptr + ex_next_offsets, mask=mask)
        ey_curr_vals = tl.load(ey_ptr + ey_curr_offsets, mask=mask)
        ey_next_vals = tl.load(ey_ptr + ey_next_offsets, mask=mask)
        
        new_hz = hz_vals - 0.7 * ((ex_next_vals - ex_curr_vals) + (ey_next_vals - ey_curr_vals))
        tl.store(hz_ptr + hz_offsets, new_hz, mask=mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid = (triton.cdiv(NX, BLOCK_I), triton.cdiv(NY, BLOCK_J))
    
    fdtd_2d_kernel[grid](
        ex, ey, hz, _fict_,
        NX, NY, TMAX,
        BLOCK_I, BLOCK_J
    )