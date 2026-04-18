import argparse
import ast
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir

parser = argparse.ArgumentParser(
    description='Language Statistics',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--language-extracts',
    type=str,
    required=True,
    help='Path to deduplicated processed files.',
)
parser.add_argument(
    '--log-file-path',
    type=str,
    default='language_statistics.log',
    help='Path to log file.',
)


def get_language_statistics(language_extracts_path: Path, plot_path: Path) -> None:
    """Statistics on extracted column Language."""
    frequency_languages: Dict[str, int] = {}
    thresholds = []
    language_thresholds: Dict[str, List] = {}

    for path in tqdm(language_extracts_path.glob('*.parquet'), desc='Iterating shards'):
        df = pd.read_parquet(path, engine='pyarrow')
        for lang in df['language']:
            lang = ast.literal_eval(lang)
            score, lang_code = lang[0], lang[1]

            frequency_languages[lang_code] = frequency_languages.get(lang_code, 0) + 1

            if lang_code not in language_thresholds:
                language_thresholds[lang_code] = []
            language_thresholds[lang_code].append(score)
            thresholds.append(score)

    logging.info(f'Total domains labelled with language: {len(thresholds)}')
    logging.info(f'Number of languages found: {len(frequency_languages)}')
    logging.info(f'Languages found: {frequency_languages.keys()}')

    with open(str(plot_path / 'frequency_languages.pkl'), 'wb') as f:
        pickle.dump(frequency_languages, f)

    with open(str(plot_path / 'language_thresholds.pkl'), 'wb') as f:
        pickle.dump(language_thresholds, f)


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    language_extracts_path = Path(args.language_extracts)
    plot_path = root / 'plots'
    plot_path.mkdir(parents=True, exist_ok=True)

    get_language_statistics(
        language_extracts_path=language_extracts_path,
        plot_path=plot_path,
    )


if __name__ == '__main__':
    main()
