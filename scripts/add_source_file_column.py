import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_scratch

parser = argparse.ArgumentParser(
    description='Add source column.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--shard-paths',
    type=str,
    default='data/test_data/data/sample/',
    help='Path to parquet files to process.',
)
parser.add_argument(
    '--output-path',
    type=str,
    default='data/test_data/data/',
    help='Path to parquet files to process.',
)
parser.add_argument(
    '--log-file-path',
    type=str,
    default='reduce_partition_size.log',
    help='Path to log file.',
)


def add_source_column(input_file: Path, output_dir: Path) -> None:
    df = pd.read_parquet(input_file)

    if '__source_file_' in df.columns:
        logging.warning(
            f'Skipping {input_file.name}: __source_file_column already included.'
        )
        return

    df['__source_file'] = input_file.name
    output_path = output_dir / input_file.name

    df.to_parquet(output_path, index=False)
    logging.info(f'Stamped {input_file} -> {output_path}')


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    scratch = get_scratch()
    source_dir = scratch / args.shard_paths
    output_dir = source_dir / 'curator_shards'
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = [p for p in source_dir.glob('*.parquet') if p.is_file()]

    logging.info(f'Found {len(parquet_files)} shard(s) in {source_dir}')

    for p_file in tqdm(parquet_files, desc='parquet files'):
        add_source_column(p_file, output_dir)

    logging.info('Source addition complete.')


if __name__ == '__main__':
    main()
