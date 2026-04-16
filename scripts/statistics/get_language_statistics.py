import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_language_stats(
    frequency_dict: dict, thresholds: list, lang_thresholds: dict, output_dir: Path
) -> None:
    """Plots language frequency and threshold distributions with explicit labels."""
    sns.set_theme(style='whitegrid')

    # 1. Bar Plot: Language Frequency
    plt.figure(figsize=(14, 7))
    lang_df = pd.DataFrame(
        list(frequency_dict.items()), columns=['Language', 'Count']
    ).sort_values('Count', ascending=False)

    ax1 = sns.barplot(
        data=lang_df,
        x='Language',
        y='Count',
        hue='Language',
        palette='viridis',
        legend=False,
    )

    ax1.set_yscale('log')
    ax1.set_title('Language Distribution (Log Scale)', fontsize=14)
    ax1.set_xlabel('Language', fontsize=12)
    ax1.set_ylabel('Count (Log)', fontsize=12)

    # Explicitly set ticks to ensure every language in the dataframe is shown
    ax1.set_xticks(range(len(lang_df['Language'])))
    ax1.set_xticklabels(lang_df['Language'], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'language_frequency.png')
    plt.close()

    # 2. Box Plot: Threshold distribution per Language
    thresh_data = []
    for lang, vals in lang_thresholds.items():
        for v in vals:
            thresh_data.append({'Language': lang, 'Threshold': v})

    thresh_df = pd.DataFrame(thresh_data)

    # Ensure the boxplot follows the same language order as the bar chart
    lang_order = lang_df['Language'].tolist()

    plt.figure(figsize=(14, 7))
    ax2 = sns.boxplot(
        data=thresh_df,
        x='Language',
        y='Threshold',
        order=lang_order,
        hue='Language',
        palette='magma',
        legend=False,
    )

    ax2.set_title('Confidence Threshold Distribution per Language', fontsize=14)
    ax2.set_xlabel('Language', fontsize=12)
    ax2.set_ylabel('Confidence Score', fontsize=12)

    # Explicitly set ticks
    ax2.set_xticks(range(len(lang_order)))
    ax2.set_xticklabels(lang_order, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'language_thresholds.png')
    plt.close()


def get_language_statistics(language_extracts_path: Path, plot_path: Path) -> None:
    """Statistics on extracted column Language."""
    frequency_languages: Dict[str, int] = {}
    thresholds = []
    language_thresholds: Dict[str, List] = {}

    for path in tqdm(language_extracts_path.glob('*.parquet'), desc='Iterating shards'):
        df = pd.read_parquet(path, engine='pyarrow')
        for lang in df['language']:
            score, lang_code = lang[0], lang[1]

            frequency_languages[lang_code] = frequency_languages.get(lang_code, 0) + 1

            if lang_code not in language_thresholds:
                language_thresholds[lang_code] = []
            language_thresholds[lang_code].append(score)
            thresholds.append(score)

    logging.info(f'Total domains labelled with language: {len(thresholds)}')
    logging.info(f'Number of languages found: {len(frequency_languages)}')
    logging.info(f'Languages found: {frequency_languages.keys()}')

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
