import argparse
import logging
import time
from pathlib import Path

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.modifiers import (
    NewlineNormalizer,
    UnicodeReformatter,
    UrlRemover,
)
from nemo_curator.stages.text.modules import Modify

from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Text cleaning pipeline.',
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
    '--files-per-partition',
    type=int,
    default=4,
    help='Filers per ray partition.',
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
    default='text_cleaning.log',
    help='Path to log file.',
)


def run_text_cleaning(
    file_paths: Path,
    output_path: Path,
    files_per_partition: int,
    num_cpus: int,
    num_gpus: int,
) -> None:
    ray_client = RayClient(num_cpus=num_cpus, num_gpus=num_gpus)
    ray_client.start()
    time.sleep(10)
    try:
        pipeline = Pipeline(name='Text cleaning pipeline.')

        reader = ParquetReader(
            file_paths=str(file_paths),
            files_per_partition=files_per_partition,
            fields=['Domain_Name', 'wet_record_txt'],
        )
        pipeline.add_stage(reader)

        pipeline.add_stage(
            Modify(modifier_fn=UnicodeReformatter(), input_fields='wet_record_txt')
        )
        pipeline.add_stage(
            Modify(modifier_fn=NewlineNormalizer(), input_fields='wet_record_txt')
        )
        pipeline.add_stage(
            Modify(modifier_fn=UrlRemover(), input_fields='wet_record_txt')
        )

        pipeline.add_stage(ParquetWriter(str(output_path)))
        results = pipeline.run()

        for task in results:
            for perf in task._stage_perf:
                logging.info(f'Stage: {perf.stage_name}')
                logging.info(f'     Duration: {perf.process_time}s')
                logging.info(f'     Items Processed: {perf.num_items_processed}')

        ray_client.stop()
    except Exception as e:
        import traceback

        logging.info(
            f'Text cleaning failed with exception: {e}\n{traceback.format_exc()}'
        )
        ray_client.stop()


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    file_paths = Path(args.file_paths)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info('Running Text Cleaning.')
    run_text_cleaning(
        file_paths=file_paths,
        output_path=output_path,
        files_per_partition=args.files_per_partition,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
    )


if __name__ == '__main__':
    main()
