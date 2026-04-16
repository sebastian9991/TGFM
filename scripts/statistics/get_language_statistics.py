import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def plot_language_stats(
    frequency_dict: dict, thresholds: list, lang_thresholds: dict, output_dir: Path
) -> None:
    """Plots language frequency and threshold distributions."""
    sns.set_theme(style='whitegrid')

    # 1. Bar Plot: Language Frequency
    plt.figure(figsize=(12, 6))
    lang_df = pd.DataFrame(
        list(frequency_dict.items()), columns=['Language', 'Count']
    ).sort_values('Count', ascending=False)
    # Use log scale for y if there's a massive disparity between languages
    ax = sns.barplot(
        data=lang_df,
        x='Language',
        y='Count',
        hue='Language',
        palette='viridis',
        legend=False,
    )
    ax.set_yscale('log')
    plt.title('Language Distribution (Log Scale)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'language_frequency.png')
    plt.close()

    # 2. Box Plot: Threshold distribution per Language
    # Convert dict to long-form DataFrame for Seaborn
    thresh_data = []
    for lang, vals in lang_thresholds.items():
        for v in vals:
            thresh_data.append({'Language': lang, 'Threshold': v})

    thresh_df = pd.DataFrame(thresh_data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=thresh_df,
        x='Language',
        y='Threshold',
        hue='Language',
        palette='magma',
        legend=False,
    )
    plt.title('Confidence Threshold Distribution per Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'language_thresholds.png')
    plt.close()


def get_language_statistics(language_extracts_path: Path, plot_path: Path) -> None:
    """Statistics on extracted column Language."""
    frequency_languages: Dict[str, int] = {}
    thresholds = []
    language_thresholds: Dict[str, List] = {}

    for path in language_extracts_path.glob('*.parquet'):
        df = pd.read_parquet(path, engine='pyarrow')
        for lang in df['languages']:
            score, lang_code = lang[0], lang[1]

            frequency_languages[lang_code] = frequency_languages.get(lang_code, 0) + 1

            if lang_code not in language_thresholds:
                language_thresholds[lang_code] = []
            language_thresholds[lang_code].append(score)
            thresholds.append(score)

    plot_language_stats(frequency_languages, thresholds, language_thresholds, plot_path)
    logging.info(f'Completed language statistics plotting. Saved to {plot_path}')


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
