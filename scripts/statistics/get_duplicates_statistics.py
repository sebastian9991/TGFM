import argparse
import logging
from pathlib import Path

import pandas as pd

from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Deduplication statistics.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--deduplication-path',
    type=str,
    required=True,
    help='Path to deduplicated processed files.',
)
parser.add_argument(
    '--log-file-path',
    type=str,
    default='deduplicate_statistics.log',
    help='Path to log file.',
)


def get_duplicates_statistics(deduplicate_path: Path) -> None:
    """Statistics on duplicates."""
    # NOTE: This may break with large parquet files.

    deduplicate_path = deduplicate_path / 'ids_to_remove' / 'FuzzyDuplicateIds'
    duplicates_df = pd.read_parquet(deduplicate_path)
    logging.info(f'Duplicate head:\n{duplicates_df.head()}')
    logging.info(
        f'Number of duplicate documents found for removal: {len(duplicates_df)}'
    )

    cc_path = deduplicate_path / 'cache' / 'ConnectedComponentsStage'
    cc_df = pd.read_parquet(cc_path)
    logging.info(f'Connected component dataframe\n: {cc_df.head()}')
    grouped_cc_df = cc_df.groupby('_duplicate_group_id')._curator_dedup_id.agg(list)
    logging.info(f'Grouped cc dataframe\n:{grouped_cc_df.head()}')
    duplicate_cluster_sizes = cc_df._duplicate_group_id.value_counts()
    logging.info(f'Cluster sizes: {duplicate_cluster_sizes}')


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    deduplicate_path = Path(args.deduplication_path)
    get_duplicates_statistics(deduplicate_path)


if __name__ == '__main__':
    main()
