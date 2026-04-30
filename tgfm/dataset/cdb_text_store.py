"""CBD text store — like EvalTextStore but for the CommonCrawl-domain dataset.

Key differences from EvalTextStore:
  - input_ids stored as int32 (xlm-roberta-base vocab is 250k, doesn't fit
    in uint16). Doubles disk vs uint16 but unavoidable for multilingual.
  - Has an explicit `has_text` boolean array tracking which domains
    actually got text from the parquet shards (vs. domains in the registry
    that were never seen in any text shard).
  - Random-write friendly: writes happen out of order during the multi-
    month iteration, with later writes overwriting earlier ones.

Schema on disk:
    <output_dir>/
        text_ids.mmap        # int32 [N, seq_len]
        text_mask.mmap       # uint8 [N, seq_len]
        has_text_mask.pt     # bool  [N] — True if domain got text from parquet
        text_meta.json       # {num_nodes, seq_len, tokenizer_name, ...}

Usage (read):
    from cbd_text_store import CBDTextStore
    store = CBDTextStore("/path/to/processed")
    feats = store.get_features(node_ids_tensor)
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


class CBDTextStore:
    """Memmap-backed text store for the CBD (CommonCrawl-domain) dataset."""

    def __init__(
        self,
        output_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        mask_rate: float = 0.15,
    ):
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.mask_rate = mask_rate
        meta_path = self.output_dir / 'text_meta.json'
        if not meta_path.exists():
            raise FileNotFoundError(
                f'No text_meta.json in {output_dir}. Run the text store builder first.'
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.num_nodes = int(self.meta['num_nodes'])
        self.seq_len = int(self.meta['seq_len'])

        special_ids = set(self.tokenizer.all_special_ids)
        self.allowed_ids = torch.tensor(
            [i for i in range(len(self.tokenizer)) if i not in special_ids],
            dtype=torch.long,
        )

        self.mmap_input_ids = np.memmap(
            self.output_dir / 'text_ids.mmap',
            dtype='int32',
            mode='r',
            shape=(self.num_nodes, self.seq_len),
        )
        self.mmap_attention_mask = np.memmap(
            self.output_dir / 'text_mask.mmap',
            dtype='uint8',
            mode='r',
            shape=(self.num_nodes, self.seq_len),
        )

        # has_text bitmap is small (N bytes), load eagerly.
        has_text_path = self.output_dir / 'has_text_mask.pt'
        if has_text_path.exists():
            self.has_text: Optional[torch.Tensor] = torch.load(
                has_text_path, weights_only=True
            )
        else:
            self.has_text = None

    def get_features(
        self, node_ids: torch.Tensor, apply_masking: bool
    ) -> Dict[str, torch.Tensor]:
        """Return tokenized features for the given node indices.

        Returns input_ids, attention_mask. Note that XLM-R does NOT use
        token_type_ids by default — we don't synthesize them.
        """
        idx = node_ids.cpu().numpy()
        input_ids = torch.from_numpy(self.mmap_input_ids[idx].astype(np.int64))
        attention_mask = torch.from_numpy(
            self.mmap_attention_mask[idx].astype(np.int64)
        )

        result = {'input_ids': input_ids, 'attention_mask': attention_mask}

        if apply_masking:
            masked_input_ids = self._create_masked_version(input_ids)
            result['masked_input_ids'] = masked_input_ids

        return result

    def _create_masked_version(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create masked version for MLM training. We use the BERT masking strategy.

        Args:
            input_ids: Original input IDs [batch_size, seq_len]

        Returns:
            Masked input IDs with some tokens replaced by [MASK]
        """
        masked_input_ids = input_ids.clone()

        # Create mask: random tokens, excluding special tokens
        prob_matrix = torch.rand(input_ids.shape)
        mask_indices = (
            (prob_matrix < self.mask_rate)
            & (input_ids != self.tokenizer.cls_token_id)
            & (input_ids != self.tokenizer.sep_token_id)
            & (input_ids != self.tokenizer.pad_token_id)
        )

        # 80% -> [MASK], 10% -> random, 10% -> original
        mask_token_indices = mask_indices & (torch.rand(input_ids.shape) < 0.8)
        random_token_indices = (
            mask_indices & ~mask_token_indices & (torch.rand(input_ids.shape) < 0.5)
        )

        allowed_ids = self.allowed_ids.to(input_ids.device)
        random_idx_in_allowed = torch.randint(
            0,
            len(allowed_ids),
            (random_token_indices.sum().item(),),
            device=input_ids.device,
        )

        # Apply masking
        masked_input_ids[mask_token_indices] = self.tokenizer.mask_token_id
        masked_input_ids[random_token_indices] = allowed_ids[random_idx_in_allowed]

        return masked_input_ids

    def __len__(self) -> int:
        return self.num_nodes


def open_writable_store(
    output_dir: Path,
    num_nodes: int,
    seq_len: int,
    tokenizer_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate empty memmap files. Returns (mmap_ids, mmap_mask, has_text bool array).

    Use this from the builder script. The mmaps are 'w+' mode so writes
    persist; multi-process writers must coordinate to avoid races.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = np.memmap(
        output_dir / 'text_ids.mmap',
        dtype='int32',
        mode='w+',
        shape=(num_nodes, seq_len),
    )
    mask = np.memmap(
        output_dir / 'text_mask.mmap',
        dtype='uint8',
        mode='w+',
        shape=(num_nodes, seq_len),
    )
    # Initialize to all zeros — domains we never write to will look like
    # all-pad sequences, which is correct ("no text" -> attention mask is 0).
    # np.memmap with mode='w+' already zero-fills on Linux; making it explicit
    # for safety on other filesystems.
    ids[:] = 0
    mask[:] = 0
    ids.flush()
    mask.flush()

    has_text = np.zeros(num_nodes, dtype=np.bool_)

    meta = {
        'num_nodes': num_nodes,
        'seq_len': seq_len,
        'tokenizer_name': tokenizer_name,
        'task_type': 'node',
        'vocab_dtype': 'int32',
    }
    with open(output_dir / 'text_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return ids, mask, has_text
