import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_scratch


def write_to_parquet_format(
    dataframe: pd.DataFrame, save_dir: Path, file_name: str
) -> None:
    """Write dataframe to parquet format under a specified file location."""
    file_path = save_dir / file_name

    logging.info(f'Saved to: {file_path}')

    dataframe.to_parquet(file_path, index=False)


def convert_to_datatrove_format(file_path: Path | str) -> pd.DataFrame:
    """Convert hierarchical parquet format to a flat datatrove parquet format,
    concatenating all 'wet_record_txt' entries per domain.
    """
    df = pd.read_parquet(file_path)

    def concatenate_texts(pages: np.ndarray) -> str:
        if not isinstance(pages, (np.ndarray, list)):
            return ''

        texts = [
            str(page.get('wet_record_txt'))
            for page in tqdm(pages, desc='Merging wet_record_txt.')
            if isinstance(page, dict) and page.get('wet_record_txt')
        ]

        return '\n\n'.join(texts)  # TODO: Is there a better document seperator?

    def extract_uris(pages: np.ndarray) -> List:
        if not isinstance(pages, (np.ndarray, list)):
            return []

        return [
            page.get('WARC_Target_URI')
            for page in tqdm(pages, desc='Merging WARC URIs')
            if isinstance(page, dict) and page.get('WARC_Target_URI')
        ]

    df['wet_record_txt'] = df['pages'].apply(concatenate_texts)
    df['WARC_Target_URIs'] = df['pages'].apply(extract_uris)

    df = df.drop(columns=['pages'])

    return df.reset_index(drop=True)


def main() -> None:
    setup_logging('convert_to_correct_parquet_format.log')
    scratch = get_scratch()
    source_dir = scratch / 'credibench_text' / 'dec'
    output_dir = source_dir / 'curator_format'
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in source_dir.glob('dec2024_wetcontent_*.parquet'):
        logging.info(f'Processing: {path.name}')

        df = convert_to_datatrove_format(path)

        new_file_name = path.stem + '_curator.parquet'
        save_dir = output_dir / new_file_name

        write_to_parquet_format(df, save_dir, new_file_name)


if __name__ == '__main__':
    main()
