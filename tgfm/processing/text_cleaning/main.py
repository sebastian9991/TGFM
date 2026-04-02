import argparse
import logging
import time

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

from tgfm.utils.path import get_scratch

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


def run_text_cleaning() -> None:
    args = parser.parse_args()
    scratch = get_scratch()
    ray_client = RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    ray_client.start()
    time.sleep(10)
    file_paths = scratch / args.file_paths

    output_path = scratch / args.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        pipeline = Pipeline(name='Text cleaning pipeline.')

        reader = ParquetReader(
            file_paths=str(file_paths),
            files_per_partition=args.files_per_partition,
            fields=['domain', 'wet_record_txt'],
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

        # # TODO: Do we need this? Likely we do not
        # word_filter = ScoreFilter(
        #     filter_obj=WordCountFilter(min_words=50, max_words=1000),
        #     text_field='wet_record_txt',
        # )

        # pipeline.add_stage(word_filter)

        pipeline.add_stage(ParquetWriter(str(output_path)))
        pipeline.run()

        ray_client.stop()
    except Exception as e:
        logging.info(f'Text cleaning failed with exception: {e}')
        ray_client.stop()


if __name__ == '__main__':
    run_text_cleaning()
