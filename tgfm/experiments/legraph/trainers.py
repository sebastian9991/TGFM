"""Training loop for LeGraph -- graph-classification variant.

Same method as the node-classification path (single shared encoder, no EMA /
teacher-student / predictor; predictive + SIGReg loss), but the data unit is
different:

    node task:  one big graph -> sample ego-nets -> views per ego-net
    graph task: many small graphs -> each graph IS the unit -> views per graph

So there is no single "full graph": RWSE is computed *per graph* (once, cached),
views/partitions are built *per graph*, and we accumulate `batch_size` graphs
into a LeJEPA batch. Evaluation pools node embeddings to a graph embedding and
runs the standard TU 10-fold linear-SVM probe over the graph labels.

NOTE: `GraphDatasetPreparer` mirrors `SubgraphPreparer` for graph classification task.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from tgfm.dataset.evaluation.linear_prob_pyg import (
    compute_node_embeddings,
    evaluate,
    evaluate_graph_classification,
)
from tgfm.dataset.pyg.data import (
    FullGraphEncodingsCache,
    GraphDatasetPreparer,
    SubgraphPreparer,
    get_dataset,
)
from tgfm.models.graphgps import GraphGPSEncoder, LinearEncoder
from tgfm.models.leJepa_loss import LeJEPALoss
from tgfm.utils.args import (
    DataArguments,
    GraphGPSArguments,
    MetaArguments,
    ModelArguments,
)
from tgfm.utils.diagnostics import collapse_diagnostics
from tgfm.utils.train import save_checkpoint
from tgfm.views.construct_views import build_views, flatten_views
from tgfm.views.prepare import compute_rwse


def compute_graph_embeddings(
    encoder: torch.nn.Module,
    dataset: InMemoryDataset,
    K: int,
    device: int,
    pool: str = 'mean',
) -> tuple[Tensor, Tensor]:
    """Encode each graph into a single vector by pooling its node embeddings.

    Reuses `compute_node_embeddings` (encoder forward over a whole graph) per
    graph, then mean/sum-pools to a graph-level embedding. Returns (Z, y) with
    Z: (G, d), y: (G,).

    TODO: this loops one graph at a time for clarity; for speed, batch graphs
    with a PyG DataLoader and pool by `batch` in one forward.
    """
    encoder.eval()
    zs: list[Tensor] = []
    ys: list[int] = []
    for graph in dataset:
        rwse = compute_rwse(graph, K=K)
        node_emb = compute_node_embeddings(
            encoder, graph, rwse=rwse, se=None, device=device
        )  # (N_i, d)
        g_emb = node_emb.sum(0) if pool == 'sum' else node_emb.mean(0)
        zs.append(g_emb.detach().cpu())
        ys.append(int(graph.y))
    return torch.stack(zs, 0), torch.tensor(ys)


def _graph_classification_transform(enabled: bool) -> Optional[T.BaseTransform]:
    """Per-graph preprocessing for graph-level tasks.

    Keep ToUndirected + AddSelfLoops (GraphGPS expects these, and a symmetric
    adjacency also keeps METIS partitioning stable). Drop NormalizeFeatures and
    RemoveIsolatedNodes from the node-classification path: row-normalizing
    categorical node labels and deleting nodes both alter graph-level structure.
    """
    if not enabled:
        return None
    return T.Compose([T.ToUndirected(), T.AddSelfLoops()])


def train_graph_task(
    model_args: ModelArguments,
    data_args: DataArguments,
    meta_args: MetaArguments,
    save_dir: Path,
) -> None:
    assert isinstance(model_args, GraphGPSArguments)
    assert data_args.task_name == 'graph'

    # 1. Load dataset (a collection of graphs, each with a graph-level label).
    if data_args.transform == True:
        transform = _graph_classification_transform(data_args.transform)
        dataset = get_dataset(
            root=str(meta_args.root_dir), name=data_args.data_name, transform=transform
        )
    else:
        dataset = get_dataset(root=str(meta_args.root_dir), name=data_args.data_name)
    # Featureless TU datasets (COLLAB, IMDB-*, REDDIT-*) ship without node
    # features. GraphGPS needs something to encode, so attach a constant feature
    # (structure then comes entirely from the RWSE PE). Swap in T.OneHotDegree
    # if you want degree-based features on the smaller IMDB/COLLAB graphs.
    # TODO: Check how this is dealt with in our literature:
    # TODO: This is an error dataset does not have attribute num_features
    if dataset.num_features == 0:
        logging.info('Dataset has no node features; adding a constant feature.')
        const = T.Constant(value=1.0)
        dataset.transform = T.Compose([const, transform]) if transform else const
    num_features = dataset[0].num_features
    num_classes = dataset.num_classes
    logging.info(
        'Loaded %s: G=%d graphs, in_dim=%d, num_classes=%d',
        data_args.data_name,
        len(dataset),
        num_features,
        num_classes,
    )

    # 2. Per-graph view preparer (precomputes RWSE once per graph).
    preparer = GraphDatasetPreparer(
        dataset=dataset,
        K=model_args.rwse_K,
        prepare_kwargs=dict(
            num_local_parts=model_args.num_local_parts,
            num_global_views=model_args.num_global_views,
            global_coverage_frac=model_args.global_coverage_frac,
            global_strategy=model_args.global_strategy,
            num_local_as_global=model_args.num_local_as_global,
        ),
    )
    logging.info('Graph-dataset preparer loaded (%d graphs).', len(dataset))

    # 3. Build dataset-specific encoders (identical to the node path).
    node_encoder = LinearEncoder(in_dim=num_features, out_dim=model_args.node_out_dim)
    pe_encoder = LinearEncoder(in_dim=model_args.rwse_K, out_dim=model_args.pe_out_dim)
    se_encoder: Optional[torch.nn.Module] = None

    encoder = GraphGPSEncoder(
        node_encoder=node_encoder,
        pe_encoder=pe_encoder,
        se_encoder=se_encoder,
        dim=model_args.dim,
        num_layers=model_args.num_layers,
        num_heads=model_args.num_heads,
        dropout=model_args.dropout,
        local_gnn_type=model_args.local_gnn_type,
        attn_type=model_args.attn_type,
        norm=model_args.norm,
    ).to(model_args.device)

    loss_fn = LeJEPALoss(
        lambd=model_args.lambd,
        num_slices=model_args.num_slices,
    ).to(model_args.device)

    optim = AdamW(
        encoder.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay
    )

    logging.info(
        'Encoder params: %.2fM',
        sum(p.numel() for p in encoder.parameters()) / 1e6,
    )

    eval_freq = model_args.eval_frequency
    eval_repeat = model_args.eval_repeat
    best_loss: float = float('inf')

    for step in tqdm(range(model_args.num_steps), desc='Training steps'):
        encoder.train()
        # Each batch is `batch_size` whole graphs; SIGReg needs B > 1 across the
        # batch dimension, so keep batch_size > 1 even though views are per graph.
        batch = preparer.sample_batch(model_args.batch_size)
        if len(batch) < model_args.batch_size:
            logging.warning(
                'Loader exhausted before filling batch; got %d / %d',
                len(batch),
                model_args.batch_size,
            )

        if len(batch) == 0:
            break

        global_views, local_views, B, V_g, V_l = flatten_views(batch)

        # Move all views to the encoder's device.
        global_views = [g.to(model_args.device) for g in global_views]
        local_views = [l.to(model_args.device) for l in local_views]

        z_global, z_local = build_views(
            encoder=encoder,
            global_views=global_views,
            local_views=local_views,
            B=B,
            V_g=V_g,
            V_l=V_l,
            edge_drop_rate=model_args.edge_drop_rate,
            feat_mask_rate=model_args.feat_mask_rate,
            augment_views=True,
        )

        out = loss_fn(z_global, z_local)

        optim.zero_grad(set_to_none=True)
        out.total.backward()
        optim.step()

        if out.total < best_loss:
            save_checkpoint(
                save_dir, step, encoder, optim, out.total, out.pred, out.sigreg
            )
            logging.info(
                f'[step {step + 1}] New best loss {out.total:4f} saved best.pt'
            )
            best_loss = out.total

        if step % model_args.log_frequency == 0 or step == model_args.num_steps - 1:
            with torch.no_grad():
                # Diagnostics on the flattened global-view embeddings.
                stats = collapse_diagnostics(z_global.reshape(-1, z_global.size(-1)))
            logging.info(
                'step=%d loss=%.4f pred=%.4f sigreg=%.4f | '
                'rank=%.0f eff_rank=%.1f trace=%.3f var[min/mean/max]=%.3f/%.3f/%.3f '
                'cos[mean/std]=%.3f/%.3f',
                step,
                out.total.item(),
                out.pred.item(),
                out.sigreg.item(),
                stats.rank,
                stats.eff_rank,
                stats.trace_cov,
                stats.var_min,
                stats.var_mean,
                stats.var_max,
                stats.cos_mean,
                stats.cos_std,
            )

        if eval_freq > 0 and step > 0 and step % eval_freq == 0:
            logging.info('step=%d running graph-classification probe...', step)
            Z, y = compute_graph_embeddings(
                encoder, dataset, K=model_args.rwse_K, device=model_args.device
            )
            results = evaluate_graph_classification(
                Z, y, repeat=eval_repeat, seed=meta_args.global_seed
            )
            logging.info(
                'step=%d prob[%s] acc=%.4f +/- %.4f over %d folds',
                step,
                results['probe_type'],
                results['mean'],
                results['std'],
                len(results['accuracies']),
            )

    # Final evaluation.
    logging.info('Final evaluation (repeat=%d)', eval_repeat)
    Z, y = compute_graph_embeddings(
        encoder, dataset, K=model_args.rwse_K, device=model_args.device
    )
    results = evaluate_graph_classification(
        Z, y, repeat=eval_repeat, seed=meta_args.global_seed
    )
    logging.info(
        'FINAL step=%d prob[%s] acc=%.4f +/- %.4f over %d folds',
        step,
        results['probe_type'],
        results['mean'],
        results['std'],
        len(results['accuracies']),
    )


def train_node_task(
    model_args: ModelArguments,
    data_args: DataArguments,
    meta_args: MetaArguments,
    save_dir: Path,
) -> None:
    assert isinstance(model_args, GraphGPSArguments)
    assert data_args.task_name == 'node'

    # 1. Load dataset
    if data_args.transform:
        transform = T.Compose(
            [
                T.NormalizeFeatures(),
                T.ToUndirected(),
                T.AddSelfLoops(),
                T.RemoveIsolatedNodes(),
            ]
        )
        dataset = get_dataset(
            root=str(meta_args.root_dir), name=data_args.data_name, transform=transform
        )
        full_data = dataset[0]
        assert full_data.has_isolated_nodes() == False
        assert full_data.has_self_loops() == True
        assert full_data.is_directed() == False
    else:
        dataset = get_dataset(root=str(meta_args.root_dir), name=data_args.data_name)
        full_data = dataset[0]
    logging.info(
        'Loaded %s: N=%d nodes, E=%d edges, in_dim=%d',
        data_args.data_name,
        full_data.num_nodes,
        full_data.num_edges,
        full_data.num_features,
    )

    # 2. Precompute RWSE on the full graph
    # TODO: We may need to cache this result if it takes too long
    cache = FullGraphEncodingsCache(full_data, K=model_args.rwse_K)
    logging.info('Precomputing RWSE (K=%d) on the full graph...', model_args.rwse_K)
    cache.precompute()
    logging.info('Computed RWSE.')

    # 3. Build the stream preparer
    preparer = SubgraphPreparer(
        full_data=full_data,
        cache=cache,
        num_neighbors=model_args.num_neighbors,
        seed_batch_size=1,
        prepare_kwargs=dict(
            num_local_parts=model_args.num_local_parts,
            num_global_views=model_args.num_global_views,
            global_coverage_frac=model_args.global_coverage_frac,
            global_strategy=model_args.global_strategy,
            num_local_as_global=model_args.num_local_as_global,
        ),
    )
    logging.info('Streaming preparer loaded.')

    # 4. Build dataset-specific encoders
    node_encoder = LinearEncoder(
        in_dim=full_data.num_features, out_dim=model_args.node_out_dim
    )
    # TODO:  Should it be an MLP?
    pe_encoder = LinearEncoder(in_dim=model_args.rwse_K, out_dim=model_args.pe_out_dim)
    se_encoder: Optional[torch.nn.Module] = None

    encoder = GraphGPSEncoder(
        node_encoder=node_encoder,
        pe_encoder=pe_encoder,
        se_encoder=se_encoder,
        dim=model_args.dim,
        num_layers=model_args.num_layers,
        num_heads=model_args.num_heads,
        dropout=model_args.dropout,
        local_gnn_type=model_args.local_gnn_type,
        attn_type=model_args.attn_type,
        norm=model_args.norm,
    ).to(model_args.device)

    loss_fn = LeJEPALoss(
        lambd=model_args.lambd,
        num_slices=model_args.num_slices,
    ).to(model_args.device)

    optim = AdamW(
        encoder.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay
    )

    # Recommended from LeJEPA paper.
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer=optim, T_0=20
    # )  # TODO: Paramaterize this.
    #
    logging.info(
        'Encoder params: %.2fM',
        sum(p.numel() for p in encoder.parameters()) / 1e6,
    )

    eval_freq = model_args.eval_frequency
    eval_repeat = model_args.eval_repeat
    best_loss: float = float('inf')
    assert cache.full_rwse is not None
    for step in tqdm(range(model_args.num_steps), desc='Training steps'):
        encoder.train()
        batch = preparer.sample_batch(model_args.batch_size)
        if len(batch) < model_args.batch_size:
            logging.warning(
                'Loader exhausted before filling batch; got %d / %d',
                len(batch),
                model_args.batch_size,
            )

        if len(batch) == 0:
            break

        global_views, local_views, B, V_g, V_l = flatten_views(batch)

        # Move all views to the encoder's device.
        global_views = [g.to(model_args.device) for g in global_views]
        local_views = [l.to(model_args.device) for l in local_views]

        z_global, z_local = build_views(
            encoder=encoder,
            global_views=global_views,
            local_views=local_views,
            B=B,
            V_g=V_g,
            V_l=V_l,
            edge_drop_rate=model_args.edge_drop_rate,
            feat_mask_rate=model_args.feat_mask_rate,
            augment_views=True,
        )

        out = loss_fn(z_global, z_local)

        optim.zero_grad(
            set_to_none=True
        )  # Is set to none correct here? Should it be set to zero by default?
        out.total.backward()
        optim.step()
        # scheduler.step()

        if out.total < best_loss:
            save_checkpoint(
                save_dir, step, encoder, optim, out.total, out.pred, out.sigreg
            )
            logging.info(
                f'[step {step + 1}] New best loss {out.total:4f} saved best.pt'
            )
            best_loss = out.total

        if step % model_args.log_frequency == 0 or step == model_args.num_steps - 1:
            with torch.no_grad():
                # Diagnostics on the flattened global-view embeddings.
                stats = collapse_diagnostics(z_global.reshape(-1, z_global.size(-1)))
            logging.info(
                'step=%d loss=%.4f pred=%.4f sigreg=%.4f | '
                'rank=%.0f eff_rank=%.1f trace=%.3f var[min/mean/max]=%.3f/%.3f/%.3f '
                'cos[mean/std]=%.3f/%.3f',
                step,
                out.total.item(),
                out.pred.item(),
                out.sigreg.item(),
                stats.rank,
                stats.eff_rank,
                stats.trace_cov,
                stats.var_min,
                stats.var_mean,
                stats.var_max,
                stats.cos_mean,
                stats.cos_std,
            )

        if model_args.eval_frequency > 0 and step > 0 and step % eval_freq == 0:
            logging.info(
                'step=%d running linear prob (repeat = %d)...', step, eval_repeat
            )
            embeddings = compute_node_embeddings(
                encoder,
                full_data,
                rwse=cache.full_rwse,
                se=cache.full_se,
                device=model_args.device,
            )
            results = evaluate(
                embeddings,
                full_data,
                repeat=eval_repeat,
                data_random_seed=meta_args.global_seed,
                dataset=data_args.data_name,
                full_eval=False,
            )
            logging.info(
                'step=%d prob[%s] acc=%.4f +/- %4.f over %d splits',
                step,
                results['probe_type'],
                results['mean'],
                results['std'],
                len(results['accuracies']),
            )

    logging.info('Final evaluation (repeat=%d)', eval_repeat)
    embeddings = compute_node_embeddings(
        encoder,
        full_data,
        rwse=cache.full_rwse,
        se=cache.full_se,
        device=model_args.device,
    )
    results = evaluate(
        embeddings,
        full_data,
        repeat=eval_repeat,
        data_random_seed=meta_args.global_seed,
        dataset=data_args.data_name,
        full_eval=True,
    )
    logging.info(
        'FINAL step=%d prob[%s] acc=%.4f +/- %4.f over %d splits',
        step,
        results['probe_type'],
        results['mean'],
        results['std'],
        len(results['accuracies']),
    )
