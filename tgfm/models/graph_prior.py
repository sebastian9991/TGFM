"""Graph-prior replacement for SIGReg in the LeGramJEPA loss.

SIGReg pushes the B rows of Z^a towards i.i.d. N(0, I). Rows of a graph batch
are not independent: connected centre nodes are expected to have correlated
embeddings. The graph prior says Z = K^{1/2} E with E i.i.d. N(0, I), where K
is a B x B kernel built from the batch's induced subgraph. Whitening with
W = K^{-1/2} removes the expected correlation, and the i.i.d. Gaussian test is
run on W Z instead of Z:

    R_G(Z^a) = SIGReg(W Z^a)             ('gsigreg')
    R_G(Z^a) = B * MMD^2(W Z^a, N(0, I))  ('gmmd', Gaussian kernel, closed form)

All modalities share W. W carries no gradient (it depends on the graph only).

Kernels, on the normalized Laplacian L = I - D^{-1/2} A D^{-1/2} (Chung's
convention: an isolated node has a zero row, so L = 0 for an edgeless batch):

    heat:  K = exp(-t L)                  W = U exp(+t Lambda / 2) U^T
    gmrf:  K = eps (L + eps I)^{-1}       W = U sqrt((Lambda + eps) / eps) U^T

The GMRF kernel is scaled by eps so that K = I for an edgeless batch. Both
kernels give W = I when t -> 0 (heat), eps -> inf (gmrf), or when the batch has
no edges; R_G is then exactly SIGReg (or the plain MMD).

Under DDP the Laplacian is the rank-local induced subgraph (cross-rank edges
are ignored), so the global W is block diagonal: each rank whitens its own
rows, then rows are gathered for the regularizer, as SIGReg already does.
"""

import math
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Source-graph adjacency and the batch's induced subgraph
# --------------------------------------------------------------------------- #
def as_edge_index(edge_index: Any) -> Tensor:
    """(2, E) long tensor from a dense edge list, SparseTensor or torch sparse."""
    if hasattr(edge_index, 'coo') and not isinstance(edge_index, Tensor):
        row, col, _ = edge_index.coo()  # torch_sparse.SparseTensor
        return torch.stack([row, col]).long().cpu()
    if isinstance(edge_index, Tensor) and edge_index.is_sparse:
        return edge_index.coalesce().indices().long().cpu()
    if isinstance(edge_index, Tensor) and edge_index.layout == torch.sparse_csr:
        return edge_index.to_sparse_coo().coalesce().indices().long().cpu()
    return torch.as_tensor(edge_index).long().cpu()


class SourceAdjacency:
    """CSR adjacency of every source graph, for induced-subgraph lookups.

    Built once from the full ``data.edge_index`` of each source dataset (the
    graph the GraphCLIP ego-subgraphs were cut from), symmetrized.
    """

    def __init__(self) -> None:
        self.rowptr: List[Tensor] = []
        self.col: List[Tensor] = []
        self.num_nodes: List[int] = []

    def add(self, edge_index: Any, num_nodes: int) -> int:
        """Register one source graph; returns its source id.

        edge_index may be a (2, E) tensor, a torch_sparse SparseTensor (how the
        GraphCLIP processed source graphs store it) or a torch sparse tensor.
        """
        ei = as_edge_index(edge_index)
        if ei.numel() > 0:
            num_nodes = max(num_nodes, int(ei.max()) + 1)
        ei = torch.cat([ei, ei.flip(0)], dim=1)
        ei = ei[:, ei[0] != ei[1]]
        key = torch.unique(ei[0] * num_nodes + ei[1])
        row, col = key // num_nodes, key % num_nodes
        counts = torch.bincount(row, minlength=num_nodes)
        rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        rowptr[1:] = counts.cumsum(0)
        self.rowptr.append(rowptr)
        self.col.append(col.to(torch.int32))
        self.num_nodes.append(num_nodes)
        return len(self.rowptr) - 1

    def neighbors(self, src: int, node: int) -> Tensor:
        rp = self.rowptr[src]
        return self.col[src][rp[node] : rp[node + 1]].long()

    def induced(self, center_id: Tensor, src_id: Tensor) -> Tensor:
        """Dense (B, B) 0/1 adjacency among the batch's centre nodes.

        Nodes from different source graphs are never connected. A node that
        appears twice in the batch is not connected to itself.
        """
        center_id = center_id.long().cpu()
        src_id = src_id.long().cpu()
        B = center_id.numel()
        A = torch.zeros(B, B)
        for s in src_id.unique().tolist():
            pos = (src_id == s).nonzero(as_tuple=True)[0]
            ids = center_id[pos]
            rp, col = self.rowptr[s], self.col[s]
            starts, ends = rp[ids], rp[ids + 1]
            deg = ends - starts
            if deg.sum() == 0:
                continue
            # Flatten all neighbour lists of the batch nodes in one go.
            owner = torch.repeat_interleave(torch.arange(len(ids)), deg)
            offs = torch.arange(int(deg.sum())) - torch.repeat_interleave(
                deg.cumsum(0) - deg, deg
            )
            nbr = col[torch.repeat_interleave(starts, deg) + offs].long()
            hit = torch.isin(nbr, ids)
            if not hit.any():
                continue
            owner, nbr = owner[hit], nbr[hit]
            # Map neighbour node ids back to every batch position holding them.
            order = torch.argsort(ids)
            sorted_ids = ids[order]
            lo = torch.searchsorted(sorted_ids, nbr, right=False)
            hi = torch.searchsorted(sorted_ids, nbr, right=True)
            reps = hi - lo
            owner = torch.repeat_interleave(owner, reps)
            k = torch.arange(int(reps.sum())) - torch.repeat_interleave(
                reps.cumsum(0) - reps, reps
            )
            tgt = order[torch.repeat_interleave(lo, reps) + k]
            A[pos[owner], pos[tgt]] = 1.0
        A = torch.maximum(A, A.T)
        A.fill_diagonal_(0.0)
        return A


def batch_edge_stats(A: Tensor) -> Dict[str, float]:
    """Density diagnostics of a (B, B) batch adjacency."""
    B = A.shape[0]
    n_edges = float(A.sum() / 2)
    return {
        'edges': n_edges,
        'density': 2 * n_edges / max(1, B * (B - 1)),
        'frac_nonisolated': float((A.sum(1) > 0).float().mean()),
        'mean_degree': float(A.sum(1).mean()),
    }


# --------------------------------------------------------------------------- #
# Kernel whitening
# --------------------------------------------------------------------------- #
def normalized_laplacian(A: Tensor) -> Tensor:
    """L = I - D^{-1/2} A D^{-1/2}, with zero rows for isolated nodes."""
    deg = A.sum(1)
    inv_sqrt = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
    L = -(inv_sqrt[:, None] * A * inv_sqrt[None, :])
    L.diagonal().add_((deg > 0).to(A.dtype))
    return L


@torch.no_grad()
def whitening_matrix(
    A: Tensor, kernel: str = 'heat', t: float = 1.0, eps: float = 1.0
) -> Optional[Tensor]:
    """W = K^{-1/2} for the batch kernel K; None means W = I (edgeless batch)."""
    if A.sum() == 0:
        return None
    L = normalized_laplacian(A.double())
    lam, U = torch.linalg.eigh(L)
    lam = lam.clamp(0.0, 2.0)
    if kernel == 'heat':
        w = torch.exp(0.5 * t * lam)
    elif kernel == 'gmrf':
        w = torch.sqrt((lam + eps) / eps)
    else:
        raise ValueError(f'graph_kernel must be heat|gmrf, got {kernel!r}')
    return (U * w) @ U.T


# --------------------------------------------------------------------------- #
# Closed-form Gaussian-kernel MMD to N(0, I)
# --------------------------------------------------------------------------- #
def gaussian_mmd_to_normal(z: Tensor, sigma2: float) -> Tensor:
    """Unbiased MMD^2 between the rows of z (B, d) and N(0, I_d).

    k(x, y) = exp(-||x - y||^2 / (2 sigma2)). Expectations against the normal
    are closed form:
        E_y k(x, y)    = (sigma2 / (sigma2 + 1))^{d/2} exp(-||x||^2 / (2 (sigma2 + 1)))
        E_{y,y'} k     = (sigma2 / (sigma2 + 2))^{d/2}
    """
    B, d = z.shape
    z = z.float()
    sq = (z * z).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * z @ z.T).clamp_min(0.0)
    kxx = torch.exp(-d2 / (2.0 * sigma2))
    pp = (kxx.sum() - kxx.diagonal().sum()) / (B * (B - 1))
    c1 = math.exp(0.5 * d * math.log(sigma2 / (sigma2 + 1.0)))
    pq = c1 * torch.exp(-sq / (2.0 * (sigma2 + 1.0))).mean()
    qq = math.exp(0.5 * d * math.log(sigma2 / (sigma2 + 2.0)))
    return pp - 2.0 * pq + qq


def gmmd(z: Tensor, bandwidth: float = 1.0) -> Tensor:
    """B * MMD^2 (test-statistic scale, like SIGReg's B * EP); sigma2 = bandwidth * d."""
    B, d = z.shape
    return B * gaussian_mmd_to_normal(z, bandwidth * d)


def apply_whitening(z: Tensor, W: Optional[Tensor]) -> Tensor:
    return z if W is None else W.to(device=z.device, dtype=z.dtype) @ z


# --------------------------------------------------------------------------- #
# Graph-clustered batches
# --------------------------------------------------------------------------- #
class ClusterBatchSampler(torch.utils.data.Sampler):
    """Batches of BFS-grown clusters of centre nodes, so batches carry edges.

    A uniformly shuffled batch of a few hundred centres drawn from graphs with
    10^5-10^6 nodes has almost no edges among its centres, and R_G then equals
    SIGReg. This sampler builds each batch from batch_size // cluster_size
    groups: pick a random seed centre, grow it by BFS over the source graph
    through centres that are in this rank's shard and not yet in the batch,
    and restart from a new seed if the BFS runs dry. Every batch has exactly
    batch_size rows (drop_last semantics); an epoch has len(shard) // batch_size
    batches, so a centre may appear in more than one batch of an epoch.
    """

    def __init__(
        self,
        adj: SourceAdjacency,
        center_id: Tensor,
        src_id: Tensor,
        batch_size: int,
        cluster_size: int,
        seed: int = 0,
    ) -> None:
        import numpy as np

        self.np = np
        self.adj = adj
        self.batch_size = batch_size
        self.cluster_size = max(1, cluster_size)
        self.seed = seed
        self.epoch = 0
        self.n = len(center_id)
        self.center_id = center_id.long().numpy()
        self.src_id = src_id.long().numpy()
        # (src, node id) -> shard position, -1 when the node is not in the shard.
        self.pos: Dict[int, 'np.ndarray'] = {}
        for s in np.unique(self.src_id).tolist():
            m = np.full(adj.num_nodes[s], -1, dtype=np.int64)
            where = np.nonzero(self.src_id == s)[0]
            m[self.center_id[where]] = where
            self.pos[s] = m
        self.rowptr = [rp.numpy() for rp in adj.rowptr]
        self.col = [c.numpy() for c in adj.col]

    def __len__(self) -> int:
        return self.n // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batch(self, rng: 'np.random.Generator') -> List[int]:
        self.np
        out: List[int] = []
        used = set()
        while len(out) < self.batch_size:
            need = min(self.cluster_size, self.batch_size - len(out))
            seed = int(rng.integers(self.n))
            if seed in used:
                continue
            group, queue = [seed], [seed]
            used.add(seed)
            while queue and len(group) < need:
                p = queue.pop(0)
                s, u = int(self.src_id[p]), int(self.center_id[p])
                nbr = self.col[s][self.rowptr[s][u] : self.rowptr[s][u + 1]]
                cand = self.pos[s][nbr]
                cand = cand[cand >= 0]
                rng.shuffle(cand)
                for q in cand.tolist():
                    if q not in used:
                        used.add(q)
                        group.append(q)
                        queue.append(q)
                        if len(group) >= need:
                            break
            out.extend(group)
        return out

    def __iter__(self) -> Iterator[List[int]]:
        rng = self.np.random.default_rng(self.seed + 7919 * self.epoch)
        self.epoch += 1
        for _ in range(len(self)):
            yield self._batch(rng)
