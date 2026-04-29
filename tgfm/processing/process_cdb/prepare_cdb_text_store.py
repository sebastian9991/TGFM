"""Stage 3 of CBD preprocessing: build the tokenized text store.

Reads per-month parquet shards, joins to the domain registry, tokenizes,
and writes a single memmap-backed text store. Months are processed in
chronological order (oldest first) so that later writes overwrite earlier
ones — this implements "most recent month wins" by construction.

Inputs:
    --input-root <dir>/
        oct/parquet_text/*.parquet
        nov/parquet_text/*.parquet
        ...
    --registry $SCRATCH/cbd/processed/domain_registry.parquet  (from stage 1)

Outputs (under --output-root):
    text_ids.mmap          # int32 [N, seq_len]
    text_mask.mmap         # uint8 [N, seq_len]
    has_text_mask.pt       # bool  [N] — True if got text
    text_meta.json
    text_build_stats.json  # processing stats

Usage:
    uv run prepare_cbd_text_store.py \
        --input-root $SCRATCH/cbd/raw \
        --registry $SCRATCH/cbd/processed/domain_registry.parquet \
        --output-root $SCRATCH/cbd/processed \
        --tokenizer xlm-roberta-base \
        --seq-len 512 \
        --batch-size 1024
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tgfm.dataset.cdb_text_store import open_writable_store
from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Prepare CDB text store.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--input-root', required=True, type=Path)
parser.add_argument('--registry', required=True, type=Path)
parser.add_argument('--output-root', required=True, type=Path)
parser.add_argument('--tokenizer', default='xlm-roberta-base')
parser.add_argument('--seq-len', type=int, default=512)
parser.add_argument(
    '--batch-size',
    type=int,
    default=1024,
    help='Tokenizer batch size affects throughput, not output.',
)
parser.add_argument('--text-column', default='wet_record_txt')
parser.add_argument('--domain-column', default='Domain_Name')
parser.add_argument('--log-file', type=str, default='prepare_text_store.log')


MONTH_ORDER = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
}


def discover_month_dirs(input_root: Path) -> List[Path]:
    """Find <input_root>/<month>/parquet_text/, sorted chronologically.

    Returns list of (path, sort_key). sort_key uses MONTH_ORDER if the
    directory name is recognized; otherwise it falls back to alphabetical.
    """
    out = []
    for d in input_root.iterdir():
        if not d.is_dir():
            continue
        parquet_dir = d / 'parquet_text'
        if not parquet_dir.exists() or not any(parquet_dir.glob('*.parquet')):
            continue
        # Try matching the directory name to a month; fall back to alpha sort.
        name_lower = d.name.lower()
        if name_lower in MONTH_ORDER:
            sort_key = MONTH_ORDER[name_lower]
        elif re.match(r'\d{6,8}', name_lower):
            # Names like "202410" or "20241201" — sort numerically.
            sort_key = int(name_lower[:8].ljust(8, '0'))
        else:
            sort_key = hash(name_lower)  # at least deterministic
            logging.warning(
                f'Unrecognized month dir name {d.name!r}; sort order may be wrong.'
            )
        out.append((parquet_dir, sort_key))

    out.sort(key=lambda t: t[1])
    if not out:
        raise FileNotFoundError(
            f'No <month>/parquet_text/*.parquet found under {input_root}'
        )
    return [path for path, _ in out]


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file)
    logging.info(f'Loading registry from {args.registry}')
    registry_df = pl.read_parquet(args.registry, columns=['node_id', 'domain'])
    num_nodes = len(registry_df)
    logging.info(f'Registry: {num_nodes:,} domains')

    logging.info('Reversing registry domains to canonical form for lookup')
    registry_df = registry_df.with_columns(
        pl.col('domain').str.split('.').list.reverse().list.join('.').alias('domain')
    )
    logging.info('Building in-memory domain->node_id lookup...')
    domain_to_id: Dict[str, int] = dict(
        zip(
            registry_df.get_column('domain').to_list(),
            registry_df.get_column('node_id').to_list(),
        )
    )
    del registry_df  # free polars frame; keep the dict
    logging.info(f'Lookup ready: {len(domain_to_id):,} entries')

    # ---- Allocate output store ----
    logging.info(f'Allocating text store at {args.output_root}')
    mmap_ids, mmap_mask, has_text = open_writable_store(
        args.output_root, num_nodes, args.seq_len, args.tokenizer
    )

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    vocab_size = tokenizer.vocab_size
    logging.info(f'Tokenizer: {args.tokenizer}, vocab={vocab_size:,}')
    if vocab_size > 2**31 - 1:
        raise ValueError(f'Vocab size {vocab_size} exceeds int32 range.')

    # ---- Process months in chronological order ----
    month_dirs = discover_month_dirs(args.input_root)
    logging.info(f'Processing {len(month_dirs)} months in order:')
    for d in month_dirs:
        logging.info(f'{d.parent.name} ({d})')

    # Stats
    stats: dict[str, Any] = {
        'months': [d.parent.name for d in month_dirs],
        'per_month': {},
        'total_rows_seen': 0,
        'total_writes': 0,
        'total_overwrites': 0,
        'domains_unmatched': 0,
        'intra_month_duplicates': 0,
    }

    for month_dir in tqdm(month_dirs, desc='Iterating months'):
        month_name = month_dir.parent.name
        logging.info(f'=== {month_name} ===')
        parquet_files = sorted(month_dir.glob('*.parquet'))
        logging.info(f'  {len(parquet_files)} parquet files')

        # Track domains seen within this month
        seen_this_month = set()
        m_stats: Dict[str, int] = {
            'rows_read': 0,
            'rows_written': 0,
            'rows_unmatched': 0,
            'intra_month_duplicates': 0,
        }

        for shard_path in tqdm(parquet_files, desc='Iterating shards'):
            df = pl.read_parquet(
                shard_path,
                columns=[args.domain_column, args.text_column],
            )
            n_rows = len(df)
            m_stats['rows_read'] += n_rows
            stats['total_rows_seen'] += n_rows

            # Filter null text — domains with no actual text in this shard.
            df = df.drop_nulls(subset=[args.text_column])
            domains = df.get_column(args.domain_column).to_list()
            texts = df.get_column(args.text_column).to_list()

            # Resolve to node ids; collect indices to write in this batch
            batch_node_ids: List[int] = []
            batch_texts: List[str] = []
            for d, t in zip(domains, texts):
                node_id = domain_to_id.get(str(d))
                if node_id is None:
                    m_stats['rows_unmatched'] += 1
                    stats['domains_unmatched'] += 1
                    continue
                if d in seen_this_month:
                    m_stats['intra_month_duplicates'] += 1
                    stats['intra_month_duplicates'] += 1
                    # Still write it (last-within-month wins, deterministic
                    # by shard read order).
                seen_this_month.add(d)
                batch_node_ids.append(node_id)
                # TODO: How do we deal with empty text nodes? Should we filter it or keep a placeholder?
                batch_texts.append(t if t is not None else '')

            # Tokenize in chunks for memory safety
            for i in tqdm(
                range(0, len(batch_texts), args.batch_size), desc='Writing chunks'
            ):
                chunk_ids = batch_node_ids[i : i + args.batch_size]
                chunk_texts = batch_texts[i : i + args.batch_size]

                tokenized = tokenizer(
                    chunk_texts,
                    max_length=args.seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='np',
                )
                ids_arr = tokenized['input_ids'].astype(np.int32)
                mask_arr = tokenized['attention_mask'].astype(np.uint8)

                # Track overwrites for stats
                for nid in chunk_ids:
                    if has_text[nid]:
                        stats['total_overwrites'] += 1
                    has_text[nid] = True

                # Fancy-index write to memmap. Numpy handles this efficiently.
                idx_arr = np.array(chunk_ids, dtype=np.int64)
                mmap_ids[idx_arr] = ids_arr
                mmap_mask[idx_arr] = mask_arr
                stats['total_writes'] += len(chunk_ids)
                m_stats['rows_written'] += len(chunk_ids)

        # End of month — flush mmap, log stats
        mmap_ids.flush()
        mmap_mask.flush()
        logging.info(
            f'  read={m_stats["rows_read"]:,} '
            f'written={m_stats["rows_written"]:,} '
            f'unmatched={m_stats["rows_unmatched"]:,} '
            f'intra-month-dups={m_stats["intra_month_duplicates"]:,}'
        )
        stats['per_month'][month_name] = m_stats

    # ---- Final writes and stats ----
    mmap_ids.flush()
    mmap_mask.flush()
    del mmap_ids, mmap_mask  # close

    n_with_text = int(has_text.sum())
    n_without = num_nodes - n_with_text
    logging.info(
        f'Coverage: {n_with_text:,}/{num_nodes:,} domains got text '
        f'({n_with_text / num_nodes:.1%}); '
        f'{n_without:,} domains have NO text'
    )

    # Save has_text mask
    has_text_t = torch.from_numpy(has_text.copy())  # copy: detach from mmap arr
    torch.save(has_text_t, args.output_root / 'has_text_mask.pt')

    # Update meta with final stats
    meta_path = args.output_root / 'text_meta.json'
    meta = json.load(open(meta_path))
    meta.update(
        {
            'n_with_text': n_with_text,
            'n_without_text': n_without,
            'total_overwrites': stats['total_overwrites'],
        }
    )
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Save full stats
    with open(args.output_root / 'text_build_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    logging.info(f'Done. Stats written to {args.output_root}/text_build_stats.json')


if __name__ == '__main__':
    main()
