import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from tgfm.utils.logger import setup_logging

MAX_BYTES = 128 * 1024 * 1024

parser = argparse.ArgumentParser(
    description='Reduce partition size.',
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
    '--max-MB',
    type=int,
    default=128,
    help='Max MB per shard. Assumed unit is MB. Recommended = 128MB.',
)
parser.add_argument(
    '--log-file-path',
    type=str,
    default='reduce_partition_size.log',
    help='Path to log file.',
)


def shard_parquet_file(
    input_file: Path, output_dir: Path, max_shard_bytes: int
) -> None:
    logging.info(f'Processing {input_file.name}...')
    max_shard_bytes = max_shard_bytes * 1024 * 1024  # convert to MB
    parquet_file = pq.ParquetFile(input_file)
    base_name = input_file.stem

    current_batches = []
    current_size = 0
    shard_index = 0

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=10000), desc='Iterating parquet file.'
    ):
        current_batches.append(batch)
        current_size += batch.nbytes

        if current_size >= max_shard_bytes:
            table = pa.Table.from_batches(current_batches)
            output_file = output_dir / f'{base_name}_shard_{shard_index:05d}.parquet'
            pq.write_table(table, output_file)
            logging.info(
                f'Wrote {output_file.name} (Uncompressed memory size: {current_size / (1024 * 1024):.2f} MB)'
            )

            shard_index += 1
            current_batches = []
            current_size = 0

    if current_batches:
        table = pa.Table.from_batches(current_batches)
        output_file = output_dir / f'{base_name}_shard_{shard_index:05d}.parquet'
        pq.write_table(table, output_file)
        logging.info(
            f'Wrote {output_file.name} (Uncompressed memory size: {current_size / (1024 * 1024):.2f} MB)'
        )


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    source_dir = Path(args.shard_paths)
    output_dir = source_dir / 'curator_shards'
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = [p for p in source_dir.glob('*.parquet') if p.is_file()]

    for p_file in tqdm(parquet_files, desc='parquet files'):
        shard_parquet_file(p_file, output_dir, args.max_MB)

    logging.info('Partitioning complete.')


if __name__ == '__main__':
    main()
