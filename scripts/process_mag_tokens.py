import argparse
import logging

from transformers import AutoTokenizer

from tgfm.dataset.mag_memmap import MAG240MMapTextStore
from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Process the MAG 100M dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--text-csv-path', type=str, required=True, help='Path to text csv.'
)
parser.add_argument(
    '--output-memmap-path', type=str, required=True, help='Path to text csv.'
)


if __name__ == '__main__':
    args = parser.parse_args()
    setup_logging('process_mag_mmap.log')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    text_store = MAG240MMapTextStore(
        csv_path=args.text_csv_path,
        output_dir=args.output_memmap_path,
        tokenizer=tokenizer,
        max_seq_len=512,
        mask_rate=0.15,
        force_recreate=False,  # Set True to rebuild from scratch
    )
    logging.info('Completed Tokenization and memmap text store construction.')
