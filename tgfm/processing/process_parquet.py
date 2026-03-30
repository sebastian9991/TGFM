from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


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
            for page in pages
            if isinstance(page, dict) and page.get('wet_record_txt')
        ]

        return '\n\n'.join(texts)  # TODO: Is there a better document seperator?

    def extract_uris(pages: np.ndarray) -> List:
        if not isinstance(pages, (np.ndarray, list)):
            return []

        return [
            page.get('WARC_Target_URI')
            for page in pages
            if isinstance(page, dict) and page.get('WARC_Target_URI')
        ]

    df['wet_record_txt'] = df['pages'].apply(concatenate_texts)
    df['WARC_Target_URIs'] = df['pages'].apply(extract_uris)

    df = df.drop(columns=['pages'])

    return df.reset_index(drop=True)
