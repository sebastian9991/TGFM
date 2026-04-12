import argparse
import logging
import time
from pathlib import Path

from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.exact.workflow import (
    ExactDeduplicationWorkflow,
)
from nemo_curator.stages.deduplication.fuzzy.workflow import (
    FuzzyDeduplicationWorkflow,
)
from nemo_curator.stages.text.deduplication.removal_workflow import (
    TextDuplicatesRemovalWorkflow,
)

from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Text Deduplication pipeline.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--file-paths',
    type=str,
    default='data/test_data/data/sample/',
    help='Path to parquet files to process.',
)
parser.add_argument(
    '--output-path',
    type=str,
    default='data/test_data/data/curated/',
    help='Path to parquet files to process.',
)
parser.add_argument(
    '--deduplication-method',
    choices=['fuzzy', 'exact'],
    default='fuzzy',
    help='Method to use for deduplication.',
)
parser.add_argument(
    '--num-cpus',
    type=int,
    default=16,
    help='Number of cpus to pass to Ray Client.',
)
parser.add_argument(
    '--num-gpus',
    type=int,
    default=4,
    help='Number of gpus to pass to Ray Client.',
)
parser.add_argument(
    '--log-file-path',
    type=str,
    default='text_deduplicate.log',
    help='Path to log file.',
)


def run_deduplication(
    file_paths: Path,
    output_path: Path,
    ids_to_remove_path: Path,
    cache_path: Path,
    num_cpus: int,
    num_gpus: int,
    deduplication_method: str,
) -> None:
    ray_client = RayClient(num_cpus=num_cpus, num_gpus=num_gpus)
    ray_client.start()
    time.sleep(10)
    ids_to_remove_path.mkdir(parents=True, exist_ok=True)
    duplicated_method = ''
    try:
        if deduplication_method == 'fuzzy':
            duplicated_method = 'FuzzyDuplicateIds'
            fuzzy_workflow = FuzzyDeduplicationWorkflow(
                input_path=str(file_paths),
                cache_path=str(cache_path),
                output_path=str(ids_to_remove_path),
                text_field='wet_record_txt',
                perform_removal=False,
                input_filetype='parquet',
                char_ngrams=24,
                num_bands=20,
                minhashes_per_band=13,
            )
            fuzzy_workflow.run()
        elif deduplication_method == 'exact':
            duplicated_method = 'ExactDuplicateIds'
            exact_workflow = ExactDeduplicationWorkflow(
                input_path=str(file_paths),
                output_path=str(ids_to_remove_path),
                text_field='text',
                assign_id=True,
                perform_removal=False,
                input_filetype='parquet',
            )
            exact_workflow.run()

        duplicated_path = ids_to_remove_path / duplicated_method
        if any(duplicated_path.glob('*.parquet')):
            removal_workflow = TextDuplicatesRemovalWorkflow(
                input_path=str(file_paths),
                ids_to_remove_path=str(duplicated_path),
                output_path=str(output_path),
                input_filetype='parquet',
                input_id_field='_curator_dedup_id',
                ids_to_remove_duplicate_id_field='_curator_dedup_id',
                id_generator_path=str(ids_to_remove_path / 'fuzzy_id_generator.json'),
            )
            removal_workflow.run()
        else:
            logging.info(f'No duplicates found, skipping removal step.')
    except ConnectionError as e:
        import traceback

        logging.info(
            f'Deduplication failed with exception: {e}\n{traceback.format_exc()}'
        )
        ray_client.stop()
    except Exception as e:
        import traceback

        logging.info(
            f'Deduplication failed with exception: {e}\n{traceback.format_exc()}'
        )
        ray_client.stop()


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    file_paths = Path(args.file_paths)
    file_paths.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = output_path / 'cache'
    cache_path.mkdir(parents=True, exist_ok=True)
    ids_to_remove_path = output_path / 'ids_to_remove'
    run_deduplication(
        file_paths=file_paths,
        output_path=output_path,
        ids_to_remove_path=ids_to_remove_path,
        cache_path=cache_path,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        deduplication_method=args.deduplication_method,
    )


if __name__ == '__main__':
    main()
