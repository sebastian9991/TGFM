import argparse
import logging
import time
from pathlib import Path

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import FastTextLangId
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.modules import ScoreFilter

from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Language Extraction pipeline.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--file-paths',
    type=str,
    required=True,
    help='Path to parquet files to process.',
)
parser.add_argument(
    '--output-path',
    type=str,
    required=True,
    help='Path to parquet files to process.',
)
parser.add_argument(
    '--fast-text-path',
    type=str,
    required=True,
    help='Path to fast-text model.',
)
parser.add_argument(
    '--files-per-partition',
    type=int,
    default=1,
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
    default=1,
    help='Number of gpus to pass to Ray Client.',
)
parser.add_argument(
    '--log-file-path',
    type=str,
    default='language_extraction.log',
    help='Path to log file.',
)


def run_language_extraction(
    file_paths: Path,
    output_path: Path,
    fast_text_path: Path,
    files_per_partition: int,
    num_cpus: int,
    num_gpus: int,
) -> None:
    ray_client = RayClient(num_cpus=num_cpus, num_gpus=num_gpus)
    ray_client.start()
    time.sleep(10)
    try:
        pipeline = Pipeline(
            name='language_identification',
            description='Identify document languages using FastText',
        )

        pipeline.add_stage(
            ParquetReader(
                file_paths=str(file_paths),
                files_per_partition=files_per_partition,
            )
        )

        # IMPORTANT: Download lid.176.bin or lid.176.ftz from https://fasttext.cc/docs/en/language-identification.html
        fasttext_model_path = str(fast_text_path / 'lid.176.bin')
        pipeline.add_stage(
            ScoreFilter(
                FastTextLangId(
                    model_path=fasttext_model_path, min_langid_score=0.3
                ),  # TODO: Does this filter out files? We don't want that we want them initially labelled.
                text_field='wet_record_txt',
                score_field='language',
            )
        )
        pipeline.add_stage(ParquetWriter(str(output_path)))
        results = pipeline.run()
        for task in results:
            for perf in task._stage_perf:
                logging.info(f'Stage: {perf.stage_name}')
                logging.info(f'     Duration: {perf.process_time}s')
                logging.info(f'     Items Processed: {perf.num_items_processed}')

    except ConnectionError as e:
        import traceback

        logging.info(
            f'Language extraction failed with exception: {e}\n{traceback.format_exc()}'
        )
        ray_client.stop()
    except Exception as e:
        import traceback

        logging.info(
            f'Language extraction failed with exception: {e}\n{traceback.format_exc()}'
        )
        ray_client.stop()


def main() -> None:
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    file_paths = Path(args.file_paths).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    fast_text_path = Path(args.fast_text_path).resolve()
    run_language_extraction(
        file_paths=file_paths,
        output_path=output_path,
        fast_text_path=fast_text_path,
        files_per_partition=args.files_per_partition,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
    )


if __name__ == '__main__':
    main()
