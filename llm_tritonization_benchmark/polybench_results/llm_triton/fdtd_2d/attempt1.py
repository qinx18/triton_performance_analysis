import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr):
    
    # Time loop - must be sequential in FDTD
    for t in range(TMAX):
        # Step 1: Set boundary condition ey[0][j] = _fict_[t]
        j_offs = tl.arange(0, 128)
        j_mask = j_offs < NY
        fict_val = tl.load(_fict_ptr + t)
        ey_boundary_offs = 0 * NY + j_offs  # row 0
        tl.store(ey_ptr + ey_boundary_offs, fict_val, mask=j_mask)
        
        # Step 2: Update ey[i][j] for i=1 to NX-1
        i_offs = tl.arange(0, 64)
        j_offs_inner = tl.arange(0, 128)
        
        for i_block in range(1, NX, 64):
            i_indices = i_block + i_offs
            i_mask = i_indices < NX
            
            for j_block in range(0, NY, 128):
                j_indices = j_block + j_offs_inner
                j_mask = j_indices < NY
                
                # Create 2D masks and indices
                i_2d = i_indices[:, None]
                j_2d = j_indices[None, :]
                mask_2d = i_mask[:, None] & j_mask[None, :]
                
                # Linear indices
                ey_idx = i_2d * NY + j_2d
                hz_idx = i_2d * NY + j_2d
                hz_prev_idx = (i_2d - 1) * NY + j_2d
                
                # Load values
                ey_val = tl.load(ey_ptr + ey_idx, mask=mask_2d, other=0.0)
                hz_val = tl.load(hz_ptr + hz_idx, mask=mask_2d, other=0.0)
                hz_prev = tl.load(hz_ptr + hz_prev_idx, mask=mask_2d, other=0.0)
                
                # Update ey
                new_ey = ey_val - 0.5 * (hz_val - hz_prev)
                tl.store(ey_ptr + ey_idx, new_ey, mask=mask_2d)
        
        # Step 3: Update ex[i][j] for j=1 to NY-1
        for i_block in range(0, NX, 64):
            i_indices = i_block + i_offs
            i_mask = i_indices < NX
            
            for j_block in range(1, NY, 128):
                j_indices = j_block + j_offs_inner
                j_mask = j_indices < NY
                
                # Create 2D masks and indices
                i_2d = i_indices[:, None]
                j_2d = j_indices[None, :]
                mask_2d = i_mask[:, None] & j_mask[None, :]
                
                # Linear indices
                ex_idx = i_2d * NY + j_2d
                hz_idx = i_2d * NY + j_2d
                hz_prev_idx = i_2d * NY + (j_2d - 1)
                
                # Load values
                ex_val = tl.load(ex_ptr + ex_idx, mask=mask_2d, other=0.0)
                hz_val = tl.load(hz_ptr + hz_idx, mask=mask_2d, other=0.0)
                hz_prev = tl.load(hz_ptr + hz_prev_idx, mask=mask_2d, other=0.0)
                
                # Update ex
                new_ex = ex_val - 0.5 * (hz_val - hz_prev)
                tl.store(ex_ptr + ex_idx, new_ex, mask=mask_2d)
        
        # Step 4: Update hz[i][j] for i=0 to NX-2, j=0 to NY-2
        for i_block in range(0, NX-1, 64):
            i_indices = i_block + i_offs
            i_mask = i_indices < (NX - 1)
            
            for j_block in range(0, NY-1, 128):
                j_indices = j_block + j_offs_inner
                j_mask = j_indices < (NY - 1)
                
                # Create 2D masks and indices
                i_2d = i_indices[:, None]
                j_2d = j_indices[None, :]
                mask_2d = i_mask[:, None] & j_mask[None, :]
                
                # Linear indices
                hz_idx = i_2d * NY + j_2d
                ex_idx = i_2d * NY + j_2d
                ex_next_idx = i_2d * NY + (j_2d + 1)
                ey_idx = i_2d * NY + j_2d
                ey_next_idx = (i_2d + 1) * NY + j_2d
                
                # Load values
                hz_val = tl.load(hz_ptr + hz_idx, mask=mask_2d, other=0.0)
                ex_val = tl.load(ex_ptr + ex_idx, mask=mask_2d, other=0.0)
                ex_next = tl.load(ex_ptr + ex_next_idx, mask=mask_2d, other=0.0)
                ey_val = tl.load(ey_ptr + ey_idx, mask=mask_2d, other=0.0)
                ey_next = tl.load(ey_ptr + ey_next_idx, mask=mask_2d, other=0.0)
                
                # Update hz
                new_hz = hz_val - 0.7 * ((ex_next - ex_val) + (ey_next - ey_val))
                tl.store(hz_ptr + hz_idx, new_hz, mask=mask_2d)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    fdtd_2d_kernel[1,](
        _fict_, ex, ey, hz,
        NX=NX, NY=NY, TMAX=TMAX
    )