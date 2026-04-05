import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_scratch

MAX_BYTES = 128 * 1024 * 1024


def shard_parquet_file(
    input_file: Path, output_dir: Path, max_shard_bytes: int
) -> None:
    logging.info(f'Processing {input_file.name}...')
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
    setup_logging('shard_parquet_files.log')
    scratch = get_scratch()
    source_dir = scratch / 'credibench_text' / 'dec'
    output_dir = source_dir / 'curator_shards'
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = [p for p in source_dir.glob('*.parquet') if p.is_file()]

    for p_file in tqdm(parquet_files, desc='parquet files'):
        shard_parquet_file(p_file, output_dir, MAX_BYTES)

    logging.info('Partitioning complete.')


if __name__ == '__main__':
    main()
