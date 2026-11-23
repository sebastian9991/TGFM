import torch

from tgfm.models.graph_prior import (
    ClusterBatchSampler,
    SourceAdjacency,
    batch_edge_stats,
    gmmd,
    normalized_laplacian,
    whitening_matrix,
)
from tgfm.models.losses.volumeloss import LeGTJEPAVolumeLoss
from tgfm.models.sigreg import SIGReg
from tgfm.utils.args import LeGTJEPAArguments


def ring(n: int) -> torch.Tensor:
    src = torch.arange(n)
    return torch.stack([src, (src + 1) % n])


def random_graph(n: int, m: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, n, (2, m), generator=g)


def dense_adj(edge_index: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    ids = ids.tolist()
    E = set(map(tuple, edge_index.T.tolist()))
    B = len(ids)
    A = torch.zeros(B, B)
    for i in range(B):
        for j in range(B):
            if ids[i] != ids[j] and ((ids[i], ids[j]) in E or (ids[j], ids[i]) in E):
                A[i, j] = 1
    return A


def test_induced_matches_bruteforce():
    adj = SourceAdjacency()
    e0, e1 = random_graph(50, 200, 0), random_graph(30, 90, 1)
    adj.add(e0, 50)
    adj.add(e1, 30)
    g = torch.Generator().manual_seed(3)
    center = torch.cat([torch.randint(0, 50, (20,), generator=g), torch.randint(0, 30, (15,), generator=g)])
    src = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(15, dtype=torch.long)])
    perm = torch.randperm(35, generator=g)
    center, src = center[perm], src[perm]
    A = adj.induced(center, src)
    ref = torch.zeros(35, 35)
    for s, e in ((0, e0), (1, e1)):
        pos = (src == s).nonzero(as_tuple=True)[0]
        ref[pos[:, None], pos[None, :]] = dense_adj(e, center[pos])
    assert torch.equal(A, ref)


def test_laplacian_edgeless_is_zero_and_W_identity():
    A = torch.zeros(6, 6)
    assert torch.equal(normalized_laplacian(A), torch.zeros(6, 6))
    assert whitening_matrix(A, 'heat', 5.0) is None
    A[0, 1] = A[1, 0] = 1
    L = normalized_laplacian(A)
    assert L[3, 3] == 0 and L[0, 0] == 1 and torch.isclose(L[0, 1], torch.tensor(-1.0))


def test_whitening_inverts_kernel():
    adj = SourceAdjacency()
    adj.add(ring(12), 12)
    A = adj.induced(torch.arange(12), torch.zeros(12, dtype=torch.long)).double()
    L = normalized_laplacian(A)
    for kernel, t, eps in (('heat', 1.0, 1.0), ('gmrf', 1.0, 0.5)):
        W = whitening_matrix(A, kernel, t, eps)
        K = torch.linalg.matrix_exp(-t * L) if kernel == 'heat' else eps * torch.linalg.inv(L + eps * torch.eye(12, dtype=torch.float64))
        # W K W = I  <=>  W = K^{-1/2}
        assert torch.allclose(W @ K @ W, torch.eye(12, dtype=torch.float64), atol=1e-8)


def test_t_to_zero_recovers_sigreg():
    adj = SourceAdjacency()
    adj.add(random_graph(64, 300), 64)
    A = adj.induced(torch.arange(64), torch.zeros(64, dtype=torch.long))
    assert A.sum() > 0
    z = torch.randn(64, 16)
    sig = SIGReg(num_slices=32)
    torch.manual_seed(0)
    base = sig(z)
    W = whitening_matrix(A, 'heat', 1e-8)
    torch.manual_seed(0)
    assert torch.allclose(sig(W.float() @ z), base, atol=1e-5)
    W = whitening_matrix(A, 'heat', 1.0)
    torch.manual_seed(0)
    assert not torch.allclose(sig(W.float() @ z), base, atol=1e-3)


def test_volume_loss_regularizers():
    args = LeGTJEPAArguments(regularizer='gsigreg', graph_kernel_t=1e-8, sigreg_num_slices=32)
    B, d = 32, 16
    out = {k: torch.randn(B, d) for k in ('z_g', 'pred_g', 'z_t', 'pred_t')}
    adj = SourceAdjacency()
    adj.add(ring(B), B)
    out['batch_adj'] = adj.induced(torch.arange(B), torch.zeros(B, dtype=torch.long))
    torch.manual_seed(0)
    g = LeGTJEPAVolumeLoss(args)(out)
    args.regularizer = 'sigreg'
    torch.manual_seed(0)
    s = LeGTJEPAVolumeLoss(args)(out)
    assert torch.allclose(g['loss'], s['loss'], atol=1e-6)
    args.regularizer = 'gmmd'
    m = LeGTJEPAVolumeLoss(args)(out)
    assert torch.isfinite(m['loss'])


def test_gmmd_prefers_normal():
    torch.manual_seed(0)
    good = gmmd(torch.randn(512, 32))
    collapsed = gmmd(torch.randn(512, 1).repeat(1, 32))
    shifted = gmmd(torch.randn(512, 32) + 2.0)
    assert good.abs() < 1.0 and collapsed > 10 * good.abs() and shifted > 10 * good.abs()


def test_cluster_sampler_has_edges():
    n = 2000
    adj = SourceAdjacency()
    adj.add(random_graph(n, 8000), n)
    center, src = torch.arange(n), torch.zeros(n, dtype=torch.long)
    sampler = ClusterBatchSampler(adj, center, src, batch_size=64, cluster_size=8, seed=0)
    batches = list(sampler)
    assert len(batches) == n // 64 and all(len(b) == 64 and len(set(b)) == 64 for b in batches)
    clustered = batch_edge_stats(adj.induced(center[batches[0]], src[batches[0]]))
    rand = batch_edge_stats(adj.induced(torch.randperm(n)[:64], src[:64]))
    assert clustered['frac_nonisolated'] > 0.8 and clustered['edges'] > 5 * max(1, rand['edges'])


def test_add_accepts_sparse_formats():
    from torch_sparse import SparseTensor

    e = random_graph(40, 150, 5)
    ref = SourceAdjacency()
    ref.add(e, 40)
    for fmt in (
        SparseTensor.from_edge_index(e, sparse_sizes=(40, 40)),
        SparseTensor.from_edge_index(e, sparse_sizes=(40, 40)).t(),
        torch.sparse_coo_tensor(e, torch.ones(e.size(1)), (40, 40)),
    ):
        adj = SourceAdjacency()
        adj.add(fmt, 40)
        assert torch.equal(adj.rowptr[0], ref.rowptr[0])
        assert torch.equal(adj.col[0], ref.col[0])
