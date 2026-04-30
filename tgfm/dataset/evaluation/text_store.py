"""Lightweight memmap text store for evaluation datasets.

Schema (one directory per dataset):
    <out_dir>/
        text_ids.mmap     # uint16 [N, seq_len]
        text_mask.mmap    # uint8  [N, seq_len]
        text_meta.json    # {num_nodes, seq_len, tokenizer_name, task_type}

Differences from the pretraining text store:
  1. No token_type_ids. All datasets here are single-sentence; token_type is
     always zero. Construct on-the-fly in `get_features` if needed.
  2. input_ids stored as uint16. BERT (30522) and DeBERTa (50265) vocabs fit.
     Halves disk vs. int32.
  3. No masking. Evaluation is inference-only.
  4. Shared across all datasets — write once, read from any eval script.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


class EvalTextStore:
    """Memmap-backed, read-only text store for evaluation."""

    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        meta_path = self.out_dir / 'text_meta.json'
        if not meta_path.exists():
            raise FileNotFoundError(
                f'No text_meta.json in {out_dir}. Run the dataset prep script first.'
            )
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.num_nodes = int(self.meta['num_nodes'])
        self.seq_len = int(self.meta['seq_len'])

        self.mmap_input_ids = np.memmap(
            self.out_dir / 'text_ids.mmap',
            dtype='uint16',
            mode='r',
            shape=(self.num_nodes, self.seq_len),
        )
        self.mmap_attention_mask = np.memmap(
            self.out_dir / 'text_mask.mmap',
            dtype='uint8',
            mode='r',
            shape=(self.num_nodes, self.seq_len),
        )

    def get_features(self, node_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return tokenized features for the given node indices.

        Returns a dict with `input_ids`, `attention_mask`, `token_type_ids`.
        All tensors are on CPU; caller does the .to(device) transfer.
        """
        idx = node_ids.cpu().numpy()
        # Fancy indexing returns a copy, so no explicit .copy() needed.
        input_ids = torch.from_numpy(self.mmap_input_ids[idx].astype(np.int64))
        attention_mask = torch.from_numpy(
            self.mmap_attention_mask[idx].astype(np.int64)
        )
        # token_type_ids is always zero for single-sentence inputs. Construct
        # on the fly — cheaper than storing 58 GB of zeros.
        token_type_ids = torch.zeros_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }

    def __len__(self) -> int:
        return self.num_nodes


def write_text_store(
    out_dir: Path,
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    task_type: str,
    batch_size: int = 1024,
) -> None:
    """Tokenize a list of texts and write them to a new memmap store.

    `texts[i]` is the text for node `i`. Length determines num_nodes.

    Args:
        out_dir: Directory to create. Will be overwritten.
        texts: List of raw text strings, indexed by node id.
        tokenizer: HF tokenizer. Must have a vocab < 65535 (fits in uint16).
        seq_len: Fixed sequence length. Longer texts are truncated.
        task_type: 'node', 'edge', or 'graph'. Stored in meta.
        batch_size: Batch size for tokenization.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = len(texts)
    vocab_size = tokenizer.vocab_size
    assert vocab_size < 65535, (
        f"Tokenizer vocab_size={vocab_size} doesn't fit in uint16. "
        f'Either switch to int32 in this module or use a smaller tokenizer.'
    )

    logging.info(
        f'Writing text store: num_nodes={num_nodes}, seq_len={seq_len}, '
        f'vocab={vocab_size}, tokenizer={tokenizer.name_or_path}'
    )

    ids_path = out_dir / 'text_ids.mmap'
    mask_path = out_dir / 'text_mask.mmap'

    mmap_ids = np.memmap(
        ids_path, dtype='uint16', mode='w+', shape=(num_nodes, seq_len)
    )
    mmap_mask = np.memmap(
        mask_path, dtype='uint8', mode='w+', shape=(num_nodes, seq_len)
    )

    n_empty = 0
    for start in range(0, num_nodes, batch_size):
        batch = texts[start : start + batch_size]
        # Replace empty strings with a placeholder so the tokenizer doesn't
        # produce just [CLS][SEP]. Track how many we substituted.
        batch_clean = []
        for t in batch:
            if not t or not t.strip():
                n_empty += 1
                batch_clean.append('[untitled]')
            else:
                batch_clean.append(t)
        tokenized = tokenizer(
            batch_clean,
            max_length=seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='np',
        )
        mmap_ids[start : start + len(batch)] = tokenized['input_ids'].astype('uint16')
        mmap_mask[start : start + len(batch)] = tokenized['attention_mask'].astype(
            'uint8'
        )

    mmap_ids.flush()
    mmap_mask.flush()
    del mmap_ids, mmap_mask

    meta = {
        'num_nodes': num_nodes,
        'seq_len': seq_len,
        'tokenizer_name': tokenizer.name_or_path,
        'task_type': task_type,
        'vocab_size': vocab_size,
        'n_empty_texts': n_empty,
    }
    with open(out_dir / 'text_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logging.info(
        f'Wrote {num_nodes} texts to {out_dir}. {n_empty} were empty and replaced.'
    )
