"""Stage 1 of CBD preprocessing: build the canonical domain registry.

Inputs:
    --input-root <dir>/                 # contains per-month subdirs
        oct/vertices.csv
        nov/vertices.csv
        dec/vertices.csv
        ...

Outputs (under --output-root):
    domain_registry.parquet
        Columns:
          node_id : int64   sequential, 0..N-1
          domain  : string  unique, sorted lexicographically
          min_ts  : int32   earliest YYYYMMDD seen
          max_ts  : int32   latest YYYYMMDD seen
          n_months: int8    in how many months it appeared

This file is the source of truth for the node id space. Downstream
stages must look up node_ids by joining against `domain`.

Usage:
    python prepare_cbd_vertices.py \
        --input-root $SCRATCH/cbd/raw \
        --output-root $SCRATCH/cbd/processed
"""

import argparse
import logging
from pathlib import Path
from typing import List

import polars as pl

parser = argparse.ArgumentParser(
    description='Prepare CDB vertices.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--input-root', required=True, type=Path)
parser.add_argument('--output-root', required=True, type=Path)
parser.add_argument('--log-file', default='prepare_vertices.log', type=str)


def discover_month_dirs(input_root: Path) -> List[Path]:
    """Return per-month subdirs that contain a vertices.csv, sorted by month name.

    The order doesn't matter here (we dedup with min/max anyway), but having a
    deterministic order makes logging easier to read.
    """
    dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    out = [d for d in dirs if (d / 'vertices.csv').exists()]
    if not out:
        raise FileNotFoundError(
            f'No <month>/vertices.csv found under {input_root}. '
            f'Expected layout: {input_root}/oct/vertices.csv, etc.'
        )
    return out


def main() -> None:
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    months = discover_month_dirs(args.input_root)
    logging.info(f'Found {len(months)} months: {[m.name for m in months]}')

    # ---- Stream-load all months as a lazy union ----
    # polars.scan_csv reads metadata only; the actual scan happens on collect/sink.
    lazy_frames = []
    per_month_counts = {}
    for month_dir in months:
        path = month_dir / 'vertices.csv'
        logging.info(f'Processing: {path}')
        # Count rows in pre-pass — useful for the overlap statistic later.
        # collect_schema is metadata-only; for row count we need to scan.
        n = pl.scan_csv(path).select(pl.len()).collect().item()
        per_month_counts[month_dir.name] = n
        logging.info(f'  {month_dir.name}: {n:,} rows in vertices.csv')

        lf = pl.scan_csv(
            path,
            schema={'domain': pl.Utf8, 'ts': pl.Int32},
        )
        lazy_frames.append(lf)

    total_in = sum(per_month_counts.values())
    logging.info(f'Total input rows across all months: {total_in:,}')

    combined = pl.concat(lazy_frames)

    # ---- Aggregate per-domain stats and assign node_ids ----
    logging.info('Aggregating per-domain stats (this is the slow step)...')
    registry = (
        combined.group_by('domain')
        .agg(
            pl.col('ts').min().alias('min_ts'),
            pl.col('ts').max().alias('max_ts'),
            pl.col('ts').n_unique().alias('n_months'),
        )
        .sort('domain')  # deterministic order; subdomains of same parent cluster
        .with_row_index(name='node_id')
        .with_columns(
            [
                pl.col('node_id').cast(pl.Int64),
                pl.col('n_months').cast(pl.Int8),
            ]
        )
        .collect(streaming=True)  # streaming = doesn't require all data in RAM
    )

    n_unique = len(registry)
    overlap_rate = 1.0 - n_unique / total_in if total_in else 0.0
    logging.info(
        f'Unique domains: {n_unique:,} '
        f'(overlap rate vs total input: {overlap_rate:.1%})'
    )

    # ---- Write registry ----
    out_path = args.output_root / 'domain_registry.parquet'
    registry.write_parquet(out_path, compression='zstd', compression_level=3)
    logging.info(f'Wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)')

    # ---- Stats summary ----
    n_months_dist = (
        registry.group_by('n_months').agg(pl.len().alias('count')).sort('n_months')
    )
    logging.info('Distribution of n_months (in how many months a domain appears):')
    for row in n_months_dist.iter_rows(named=True):
        pct = row['count'] / n_unique * 100
        logging.info(
            f'  n_months={row["n_months"]:2d}: {row["count"]:>12,}  ({pct:5.2f}%)'
        )


if __name__ == '__main__':
    main()
