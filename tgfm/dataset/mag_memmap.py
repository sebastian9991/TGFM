import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase


class MAG240MMapTextStore:
    """Memory-mapped text storage for graph nodes.
    Stores tokenized text (input_ids, attention_mask, token_type_ids) on disk
    and provides efficient random access during training.
    """

    def __init__(
        self,
        output_dir: str,
        tokenizer: PreTrainedTokenizerBase,
        csv_path: Optional[str] = None,
        max_seq_len: int = 512,
        mask_rate: float = 0.15,
        force_recreate: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_rate = mask_rate
        # Make sure random doesn't use special ID:
        special_ids = set(self.tokenizer.all_special_ids)
        self.allowed_ids = torch.tensor(
            [i for i in range(len(self.tokenizer)) if i not in special_ids],
            dtype=torch.long,
        )

        # Define paths for each component
        self.paths = {
            'input_ids': self.output_dir / 'input_ids.mmap',
            'attention_mask': self.output_dir / 'attention_mask.mmap',
            'token_type_ids': self.output_dir / 'token_type_ids.mmap',
            'metadata': self.output_dir / 'metadata.npz',
        }

        # Check if already created
        if self._exists() and not force_recreate:
            logging.info('Memory-mapped files already exist. Loading metadata...')
            self._load_metadata()
        else:
            assert csv_path is not None
            self.csv_path = Path(csv_path)
            logging.info('Creating memory-mapped files...')
            self._create_mmap_features()

        # Open memory-mapped arrays for reading
        self._open_mmaps()

    def _exists(self) -> bool:
        """Check if all required files exist."""
        return all(p.exists() for p in self.paths.values())

    def _count_nodes(self) -> int:
        """Count total nodes in CSV."""
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)

    def _create_mmap_features(self) -> None:
        """Create memory-mapped arrays from CSV."""
        logging.info(f'Counting nodes in {self.csv_path}...')
        num_nodes = self._count_nodes()
        logging.info(f'Found {num_nodes:,} nodes')

        # Create memory-mapped arrays
        mmap_input_ids = np.memmap(
            self.paths['input_ids'],
            dtype='int32',
            mode='w+',
            shape=(num_nodes, self.max_seq_len),
        )

        mmap_attention_mask = np.memmap(
            self.paths['attention_mask'],
            dtype='int8',
            mode='w+',
            shape=(num_nodes, self.max_seq_len),
        )

        mmap_token_type_ids = np.memmap(
            self.paths['token_type_ids'],
            dtype='int8',
            mode='w+',
            shape=(num_nodes, self.max_seq_len),
        )

        logging.info('Tokenizing and writing to disk...')
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            batch_size = 1000
            batch_indices = []
            batch_texts = []

            count_misses = 0

            for row in tqdm(reader, total=num_nodes, desc='Processing text'):
                idx = int(row['idx'])
                title = row.get('title', '').strip()
                abstract = row.get('abstract', '').strip()
                if title == '' and abstract == '':
                    count_misses += 1

                # Format text
                text = f'{title}. {abstract}' if title or abstract else ''

                batch_indices.append(idx)
                batch_texts.append(text)

                # Process batch
                if len(batch_texts) >= batch_size:
                    self._write_batch(
                        batch_indices,
                        batch_texts,
                        mmap_input_ids,
                        mmap_attention_mask,
                        mmap_token_type_ids,
                    )
                    batch_indices = []
                    batch_texts = []

            # Process remaining
            if batch_texts:
                self._write_batch(
                    batch_indices,
                    batch_texts,
                    mmap_input_ids,
                    mmap_attention_mask,
                    mmap_token_type_ids,
                )

        # Flush and close
        logging.info('Flushing and close mmaps.')
        mmap_input_ids.flush()
        mmap_attention_mask.flush()
        mmap_token_type_ids.flush()

        del mmap_input_ids
        del mmap_attention_mask
        del mmap_token_type_ids

        # Save metadata
        logging.info('Saving metadata.')
        np.savez(
            self.paths['metadata'],
            num_nodes=num_nodes,
            max_seq_len=self.max_seq_len,
        )

        logging.info(f'Found {count_misses} empty titles and abstracts.')
        logging.info(f'Created memory-mapped arrays in {self.output_dir}')

    def _write_batch(
        self,
        indices: List,
        texts: List,
        mmap_input_ids: np.ndarray,
        mmap_attention_mask: np.ndarray,
        mmap_token_type_ids: np.ndarray,
    ) -> None:
        """Tokenize and write a batch to memory-mapped arrays."""
        # Tokenize batch
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='np',
        )

        # Write to mmaps
        for i, idx in enumerate(indices):
            mmap_input_ids[idx] = tokenized['input_ids'][i]
            mmap_attention_mask[idx] = tokenized['attention_mask'][i]

            # Handle token_type_ids (may not exist for all tokenizers)
            if 'token_type_ids' in tokenized:
                mmap_token_type_ids[idx] = tokenized['token_type_ids'][i]
            else:
                mmap_token_type_ids[idx] = 0  # Default to zeros

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        metadata = np.load(self.paths['metadata'])
        self.num_nodes = int(metadata['num_nodes'])
        self.max_seq_len = int(metadata['max_seq_len'])
        logging.info(
            f'Loaded metadata: {self.num_nodes:,} nodes, max_seq_len={self.max_seq_len}'
        )

    def _open_mmaps(self) -> None:
        """Open memory-mapped arrays for reading."""
        if not hasattr(self, 'num_nodes'):
            self._load_metadata()

        self.mmap_input_ids = np.memmap(
            self.paths['input_ids'],
            dtype='int32',
            mode='r',
            shape=(self.num_nodes, self.max_seq_len),
        )

        self.mmap_attention_mask = np.memmap(
            self.paths['attention_mask'],
            dtype='int8',
            mode='r',
            shape=(self.num_nodes, self.max_seq_len),
        )

        self.mmap_token_type_ids = np.memmap(
            self.paths['token_type_ids'],
            dtype='int8',
            mode='r',
            shape=(self.num_nodes, self.max_seq_len),
        )

    def get_features(
        self, node_ids: torch.Tensor, apply_masking: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Get tokenized features for a batch of nodes.

        Args:
            node_ids: Tensor of node indices
            apply_masking: Whether to create masked version for MLM

        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids,
            and optionally masked_input_ids
        """
        # Convert to numpy
        node_ids_np = node_ids.cpu().numpy()

        # Fetch from memory-mapped arrays
        # TODO: Check the dimensions on this
        input_ids = torch.from_numpy(self.mmap_input_ids[node_ids_np].copy())
        attention_mask = torch.from_numpy(self.mmap_attention_mask[node_ids_np].copy())
        token_type_ids = torch.from_numpy(self.mmap_token_type_ids[node_ids_np].copy())

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }

        # Optionally create masked version
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
            random_token_indices.sum().item(),
            device=input_ids.device,
        )

        # Apply masking
        masked_input_ids[mask_token_indices] = self.tokenizer.mask_token_id
        masked_input_ids[random_token_indices] = allowed_ids[random_idx_in_allowed]

        return masked_input_ids

    def __len__(self) -> int:
        """Return number of nodes."""
        return self.num_nodes
