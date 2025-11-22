import torch

def s1221_pytorch(a, b):
    """
    Corrected PyTorch implementation of TSVC s1221

    Original C code:
    for (int i = 4; i < LEN_1D; i++) {
        b[i] = b[i - 4] + a[i];
    }

    Dependency pattern (stride-4):
    - Chain 0: b[4], b[8], b[12], b[16], ... (each depends on previous in chain)
    - Chain 1: b[5], b[9], b[13], b[17], ...
    - Chain 2: b[6], b[10], b[14], b[18], ...
    - Chain 3: b[7], b[11], b[15], b[19], ...

    The 4 chains are INDEPENDENT of each other, so can be processed in parallel.
    Within each chain, it's a cumulative sum pattern:

    b[4] = b[0] + a[4]
    b[8] = b[4] + a[8] = b[0] + a[4] + a[8]
    b[12] = b[8] + a[12] = b[0] + a[4] + a[8] + a[12]

    → b[chain_i] = b[chain_base] + cumsum(a[chain])
    """
    a = a.contiguous()
    b = b.contiguous()

    n = b.shape[0]

    if n < 5:
        return b

    # Process 4 independent chains
    for chain_id in range(4):
        # Indices in this chain: chain_id, chain_id+4, chain_id+8, ...
        chain_indices = torch.arange(chain_id + 4, n, 4, device=b.device)

        if len(chain_indices) == 0:
            continue

        # Base value (never modified)
        base_val = b[chain_id]

        # Values to accumulate
        a_chain = a[chain_indices]

        # Cumulative sum: b[i] = base + cumsum(a[chain])
        b[chain_indices] = base_val + torch.cumsum(a_chain, dim=0)

    return b
