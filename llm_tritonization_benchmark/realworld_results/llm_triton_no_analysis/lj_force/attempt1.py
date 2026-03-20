import triton
import triton.language as tl
import torch

@triton.jit
def lj_force_kernel(
    f_ptr, neighbors_ptr, numneigh_ptr, pos_ptr,
    cutforcesq_val, epsilon_val, sigma6_val,
    MAXNEIGHS: tl.constexpr, NLOCAL: tl.constexpr, PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current CTA's starting atom index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    atom_ids = block_start + offsets
    atom_mask = atom_ids < NLOCAL
    
    # Skip if no valid atoms in this CTA
    if tl.sum(atom_mask) > 0:
        # Clear forces for atoms in this block
        for dim in range(PAD):
            force_offsets = atom_ids * PAD + dim
            tl.store(f_ptr + force_offsets, 0.0, mask=atom_mask)
        
        # Process each atom in the block
        for atom_idx in range(BLOCK_SIZE):
            i = block_start + atom_idx
            if i < NLOCAL:
                # Load atom i's position
                xtmp = tl.load(pos_ptr + i * PAD + 0)
                ytmp = tl.load(pos_ptr + i * PAD + 1) 
                ztmp = tl.load(pos_ptr + i * PAD + 2)
                
                # Initialize force accumulators
                fix = 0.0
                fiy = 0.0
                fiz = 0.0
                
                # Get number of neighbors for atom i
                num_neighbors = tl.load(numneigh_ptr + i)
                
                # Process neighbors in blocks
                for k_start in range(0, num_neighbors, BLOCK_SIZE):
                    k_offsets = k_start + offsets
                    k_mask = k_offsets < num_neighbors
                    
                    if tl.sum(k_mask) > 0:
                        # Load neighbor indices
                        neighbor_addrs = i * MAXNEIGHS + k_offsets
                        j_indices = tl.load(neighbors_ptr + neighbor_addrs, mask=k_mask, other=0)
                        
                        # Load neighbor positions
                        jx = tl.load(pos_ptr + j_indices * PAD + 0, mask=k_mask, other=0.0)
                        jy = tl.load(pos_ptr + j_indices * PAD + 1, mask=k_mask, other=0.0)
                        jz = tl.load(pos_ptr + j_indices * PAD + 2, mask=k_mask, other=0.0)
                        
                        # Compute distance vectors
                        delx = xtmp - jx
                        dely = ytmp - jy
                        delz = ztmp - jz
                        rsq = delx * delx + dely * dely + delz * delz
                        
                        # Apply cutoff and valid neighbor mask
                        cutoff_mask = rsq < cutforcesq_val
                        valid_mask = k_mask & cutoff_mask
                        
                        # Compute LJ force for valid interactions
                        sr2 = 1.0 / rsq
                        sr6 = sr2 * sr2 * sr2 * sigma6_val
                        force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon_val
                        
                        # Mask out invalid interactions
                        force = tl.where(valid_mask, force, 0.0)
                        
                        # Accumulate forces
                        fix += tl.sum(delx * force)
                        fiy += tl.sum(dely * force)
                        fiz += tl.sum(delz * force)
                
                # Store final forces
                tl.store(f_ptr + i * PAD + 0, tl.load(f_ptr + i * PAD + 0) + fix)
                tl.store(f_ptr + i * PAD + 1, tl.load(f_ptr + i * PAD + 1) + fiy)
                tl.store(f_ptr + i * PAD + 2, tl.load(f_ptr + i * PAD + 2) + fiz)

def lj_force_triton(f, neighbors, numneigh, pos, cutforcesq_val, epsilon_val, sigma6_val, MAXNEIGHS, NLOCAL, PAD):
    BLOCK_SIZE = 128
    grid_size = triton.cdiv(NLOCAL, BLOCK_SIZE)
    
    lj_force_kernel[(grid_size,)](
        f, neighbors, numneigh, pos,
        cutforcesq_val, epsilon_val, sigma6_val,
        MAXNEIGHS, NLOCAL, PAD, BLOCK_SIZE
    )