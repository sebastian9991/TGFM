"""Stage 2 of CBD preprocessing: build the deduplicated edge index.

Inputs:
    --input-root <dir>/                 # contains per-month subdirs
        oct/edges.csv
        nov/edges.csv
        ...
    --registry <dir>/domain_registry.parquet  (from stage 1)

Outputs (under --output-root):
    edge_index.pt
        torch.LongTensor [2, E] — int64 source/dest node ids
        Edges with src or dst not in the registry are DROPPED.
        Duplicate (src, dst) pairs across months are deduplicated.
        Direction is preserved (src->dst), no automatic undirecting.

    graph.pt
        PyG Data with:
            edge_index : [2, E] int64
            num_nodes  : int (= rows in registry)
            domain     : list[str] of length num_nodes (kept for debugging)
            min_ts     : int32 [num_nodes]
            max_ts     : int32 [num_nodes]
            task_type  : 'node'  (placeholder; downstream sets this)

Usage:
    python prepare_cbd_edges.py \
        --input-root $SCRATCH/cbd/raw \
        --output-root $SCRATCH/cbd/processed \
        --registry $SCRATCH/cbd/processed/domain_registry.parquet
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Prepare CDB edges.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--input-root', required=True, type=Path)
parser.add_argument('--output-root', required=True, type=Path)
parser.add_argument('--registry', required=True, type=Path)
parser.add_argument('--log-file', default='prepare_edges.log', type=str)


def discover_month_dirs(input_root: Path) -> List[Path]:
    dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    out = [d for d in dirs if (d / 'edges.csv').exists()]
    if not out:
        raise FileNotFoundError(f'No <month>/edges.csv found under {input_root}.')
    return out


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file)

    args.output_root.mkdir(parents=True, exist_ok=True)
    months = discover_month_dirs(args.input_root)
    logging.info(f'Found {len(months)} months: {[m.name for m in months]}')

    logging.info(f'Loading registry from {args.registry}')
    registry = pl.read_parquet(args.registry, columns=['node_id', 'domain'])
    n_nodes = len(registry)
    logging.info(f'Registry has {n_nodes:,} domains')

    resolved_edges_per_month: List[pl.DataFrame] = []
    total_in = 0
    total_dropped = 0

    for month_dir in months:
        path = month_dir / 'edges.csv'
        n_in = pl.scan_csv(path).select(pl.len()).collect().item()
        total_in += n_in
        logging.info(f'  {month_dir.name}: {n_in:,} rows')

        # Two joins: src→node_id, then dst→node_id. Inner joins drop unknown.
        edges = (
            pl.scan_csv(
                path,
                schema={'src': pl.Utf8, 'dst': pl.Utf8, 'ts': pl.Int32},
            )
            .join(
                registry.lazy().rename({'domain': 'src', 'node_id': 'src_id'}),
                on='src',
                how='inner',
            )
            .join(
                registry.lazy().rename({'domain': 'dst', 'node_id': 'dst_id'}),
                on='dst',
                how='inner',
            )
            .select(['src_id', 'dst_id'])
            .collect(streaming=True)
        )
        n_resolved = len(edges)
        n_dropped = n_in - n_resolved
        total_dropped += n_dropped
        logging.info(
            f'    resolved: {n_resolved:,}, dropped (unknown endpoint): '
            f'{n_dropped:,} ({n_dropped / n_in * 100:.2f}%)'
        )
        resolved_edges_per_month.append(edges)

    logging.info('Concatenating and deduplicating edges across all months...')
    all_edges = pl.concat(resolved_edges_per_month)
    n_pre_dedup = len(all_edges)
    deduped = all_edges.unique(subset=['src_id', 'dst_id'], maintain_order=False)
    n_post_dedup = len(deduped)
    logging.info(
        f'Edges: {total_in:,} input -> {total_in - total_dropped:,} resolved -> '
        f'{n_post_dedup:,} deduped '
        f'(dup rate within resolved: {1 - n_post_dedup / max(n_pre_dedup, 1):.1%})'
    )

    src_arr = deduped.get_column('src_id').to_numpy()
    dst_arr = deduped.get_column('dst_id').to_numpy()
    edge_index = (
        torch.from_numpy(
            # Stack as [2, E]
            np.stack([src_arr, dst_arr], axis=0)
        )
        .to(torch.int64)
        .contiguous()
    )

    logging.info(f'Shape of edge_index: {edge_index.shape}')

    edge_index_path = args.output_root / 'edge_index.pt'
    torch.save(edge_index, edge_index_path)
    logging.info(
        f'Wrote {edge_index_path} '
        f'({edge_index_path.stat().st_size / 1e9:.2f} GB, '
        f'shape={list(edge_index.shape)})'
    )

    full_registry = pl.read_parquet(args.registry)
    data = Data(
        edge_index=edge_index,
        num_nodes=n_nodes,
    )
    data.min_ts = torch.from_numpy(full_registry.get_column('min_ts').to_numpy()).to(
        torch.int32
    )
    data.max_ts = torch.from_numpy(full_registry.get_column('max_ts').to_numpy()).to(
        torch.int32
    )
    data.task_type = 'node'

    graph_path = args.output_root / 'graph.pt'
    torch.save(data, graph_path)
    logging.info(f'Wrote {graph_path} ({graph_path.stat().st_size / 1e9:.2f} GB)')


if __name__ == '__main__':
    main()
