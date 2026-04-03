import argparse
import logging

from nemo_curator.utils.split_large_files import split_parquet_file_by_size

from tgfm.utils.path import get_scratch

parser = argparse.ArgumentParser(
    description='Parquet Sharding.',
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
    '--target-size-mb',
    type=int,
    default=500,
    help=' per ray partition.',
)


def run_file_splitting() -> None:
    args = parser.parse_args()
    scratch = get_scratch()
    file_paths = scratch / args.file_paths

    output_path = scratch / args.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        for path in file_paths.glob('dec2024_wetcontent_*.parquet'):
            split_parquet_file_by_size.remote(
                input_file=str(path),
                output_path=str(output_path),
                target_size_mb=args.target_size_mb,
            )
    except Exception as e:
        import traceback

        logging.info(
            f'Text cleaning failed with exception: {e}\n{traceback.format_exc()}'
        )


if __name__ == '__main__':
    run_file_splitting()
