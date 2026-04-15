import argparse
import logging
from pathlib import Path

import pandas as pd
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.tasks.document import DocumentBatch

from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Deduplication statistics.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--input-path',
    type=str,
    required=True,
    help='Path to pre-processed files.',
)
parser.add_argument(
    '--deduplication-path',
    type=str,
    required=True,
    help='Path to deduplicated processed files.',
)
parser.add_argument(
    '--display-example',
    action='store_true',
    help='Whether to display example duplicated text.',
)
parser.add_argument(
    '--block-size',
    type=str,
    default='128MiB',
    help='Block size used in deduplication.',
)
parser.add_argument(
    '--log-file-path',
    type=str,
    default='deduplicate_statistics.log',
    help='Path to log file.',
)


def get_duplicates_statistics(
    input_path: Path,
    deduplicate_path: Path,
    block_size: str,
    get_examples: bool = False,
) -> None:
    """Statistics on duplicates."""
    # NOTE: This may break with large parquet files.

    duplicates_path = deduplicate_path / 'ids_to_remove' / 'FuzzyDuplicateIds'
    duplicates_df = pd.read_parquet(duplicates_path)
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

    # As an example let's look at the group with the largest number of duplicates
    largest_duplicate_cluster = grouped_cc_df.loc[duplicate_cluster_sizes.index[0]]

    # number of docs in the removal list from this group
    docs_to_remove_in_group = duplicates_df._curator_dedup_id.isin(
        largest_duplicate_cluster
    ).sum()

    logging.info(
        f'Number of documents in the duplicate group: {len(largest_duplicate_cluster)}'
    )
    logging.info(
        f'Number of documents in the removal list from the same group: {docs_to_remove_in_group}'
    )
    assert docs_to_remove_in_group == (len(largest_duplicate_cluster) - 1)  # noqa: S101

    if get_examples:

        class CustomMergeStage(ProcessingStage[DocumentBatch, DocumentBatch]):
            """Warning: This should not be attempted with large connected components results.
            A small stage that merges the input data (using the id's generated) with the connected components result.
            Works because CC results are small enough to fit per batch.
            """

            resources = Resources(cpus=1.0)

            def process(self, batch: DocumentBatch) -> DocumentBatch:
                df = batch.to_pandas().merge(
                    cc_df, how='inner', on=[CURATOR_DEDUP_ID_STR]
                )
                return DocumentBatch(
                    task_id=batch.task_id,
                    dataset_name=batch.dataset_name,
                    data=df,
                    _stage_perf=batch._stage_perf,
                )

        pipeline = Pipeline(
            name='Explore duplicates',
            stages=[
                ParquetReader(
                    file_paths=str(input_path / 'cleaned_text'),
                    blocksize=block_size,
                    _assign_ids=True,
                ),
                CustomMergeStage(),
            ],
        )
        from nemo_curator.stages.deduplication.id_generator import (
            create_id_generator_actor,
            kill_id_generator_actor,
        )

        try:
            create_id_generator_actor(
                filepath=str(
                    deduplicate_path / 'ids_to_remove' / 'fuzzy_id_generator.json'
                ),
            )
            merged_results = pipeline.run()
            merged_df = pd.concat(
                [batch.to_pandas() for batch in merged_results]
            ).sort_values('_duplicate_group_id')
        finally:
            kill_id_generator_actor()

        logging.info(
            f'Merged_df in largest duplicate cluster\n: {merged_df[merged_df._curator_dedup_id.isin(largest_duplicate_cluster)]}'
        )

        duplicates = merged_df[
            merged_df._curator_dedup_id.isin(
                grouped_cc_df.loc[duplicate_cluster_sizes.index[0]]
            )
        ]
        logging.info(f'Duplicates dataframe in largest cluster\n: {duplicates}')
        logging.info(f'\nDocument1\n----------\n{duplicates.iloc[0].text}')
        logging.info(f'\nDocument2\n----------\n{duplicates.iloc[1].text}')


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    input_path = Path(args.input_path)
    deduplicate_path = Path(args.deduplication_path)

    get_duplicates_statistics(
        input_path=input_path,
        deduplicate_path=deduplicate_path,
        block_size=args.block_size,
        get_examples=args.display_example,
    )


if __name__ == '__main__':
    main()
