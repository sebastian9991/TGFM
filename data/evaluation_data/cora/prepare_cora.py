"""Prepare the Cora dataset for UniGraph evaluation.

Joins OFA's `cora_orig` bundle, which ships:
  - cora.content / cora.cites  -- the standard 2708-node Planetoid Cora
  - mccallum/cora/papers       -- 52535-row index: paper_id, filename, citation
  - mccallum/cora/extractions/ -- one MIME-like file per paper with Title/Abstract

The 2708 paper IDs in cora.content are a subset of the IDs in McCallum's
`papers` file. We use that index to find each paper's extraction file and
pull out the title + abstract.

Expected layout under --raw-dir:

    <raw-dir>/
        cora.content
        cora.cites
        mccallum/
            cora/
                papers
                extractions/
                    file:##....ps        # one file per paper
                    ...

Run:
    python prepare_cora.py \
        --raw-dir ~/scratch/evaluation/cora/cora_orig \
        --out-dir datasets/cora \
        --tokenizer bert-base-uncased \
        --seq-len 512 \
        --seed 0
"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer

from tgfm.dataset.evaluation.text_store import write_text_store
from tgfm.utils.logger import setup_logging

parser = argparse.ArgumentParser(
    description='Prepare cora dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--raw-dir',
    required=True,
    type=Path,
    help='Directory containing cora.content, cora.cites, '
    'and mccallum/cora/{papers,extractions/}.',
)
parser.add_argument('--out-dir', required=True, type=Path)
parser.add_argument('--tokenizer', default='bert-base-uncased')
parser.add_argument('--seq-len', type=int, default=128)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log-file', type=str, default='prepare_cora.log')


# TODO: How did they get this??
CORA_LABEL_NAMES = [
    'Case_Based',
    'Genetic_Algorithms',
    'Neural_Networks',
    'Probabilistic_Methods',
    'Reinforcement_Learning',
    'Rule_Learning',
    'Theory',
]

CORA_LABEL_TEXTS = [
    'case based methods',
    'genetic algorithms',
    'neural networks',
    'probabilistic methods',
    'reinforcement learning',
    'rule learning',
    'theory',
]


# ----------------------------------------------------------------------------
# Parsing the Planetoid-style cora.content / cora.cites
# ----------------------------------------------------------------------------
def parse_cora_content(path: Path) -> Tuple[List[str], torch.Tensor, Dict[str, int]]:
    r"""Parse cora.content. Returns (paper_ids, y, id_to_idx).

    Each line: <paper_id>\\t<1433 binary features>\\t<label_name>.
    """
    label2idx = {name: i for i, name in enumerate(CORA_LABEL_NAMES)}
    paper_ids: List[str] = []
    labels: List[int] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            paper_id = parts[0]
            label_name = parts[-1]
            if label_name not in label2idx:
                raise ValueError(f'Unknown label {label_name!r} in {path}')
            paper_ids.append(paper_id)
            labels.append(label2idx[label_name])

    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    if len(id_to_idx) != len(paper_ids):
        raise ValueError(f'Duplicate paper_ids in {path}')
    y = torch.tensor(labels, dtype=torch.long)
    return paper_ids, y, id_to_idx


def parse_cora_cites(path: Path, id_to_idx: Dict[str, int]) -> torch.Tensor:
    """Parse cora.cites. README states each line is "<cited_id> <citing_id>",
    i.e. the link direction is from column 2 to column 1.

    For undirected use this doesn't matter, but we follow the README so that
    a future directed-graph user gets correct semantics.

    Returns edge_index with both directions (undirected) of shape [2, 2*E].
    """
    src_list: List[int] = []
    dst_list: List[int] = []
    skipped = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 2:
                skipped += 1
                continue
            cited_id, citing_id = parts
            if cited_id not in id_to_idx or citing_id not in id_to_idx:
                skipped += 1
                continue
            # Edge: citing -> cited (per README, direction is right-to-left)
            src_list.append(id_to_idx[citing_id])
            dst_list.append(id_to_idx[cited_id])
    if skipped:
        logging.warning(f'Skipped {skipped} citation lines (malformed or unknown ids)')

    # Undirected: include both (u, v) and (v, u)
    # TODO: Take note of this. Perhaps remove it, we only really need one direction.
    edge_index = torch.tensor(
        [src_list + dst_list, dst_list + src_list], dtype=torch.long
    )
    return edge_index


# ----------------------------------------------------------------------------
# Joining to McCallum extractions
# ----------------------------------------------------------------------------
def parse_mccallum_papers(path: Path) -> Dict[str, List[str]]:
    r"""Build paper_id -> [filenames] map from mccallum/cora/papers.

    Format: <paper_id>\r\t<filename>\r\t<citation_string>
    A paper_id can appear multiple times with different filenames (different
    postscript URLs for the same paper). We keep all of them and try them
    in order at extraction time.
    """
    pid_to_files: Dict[str, List[str]] = defaultdict(list)
    skipped = 0
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                skipped += 1
                continue
            paper_id, filename = parts[0], parts[1]
            pid_to_files[paper_id].append(filename)
    if skipped:
        logging.warning(f'Skipped {skipped} malformed lines in {path}')
    logging.info(
        f'Loaded {len(pid_to_files):,} unique paper_ids from {path} '
        f'(total entries: {sum(len(v) for v in pid_to_files.values()):,})'
    )
    return pid_to_files


# Field names that start a new logical line in the MIME-like extraction format.
# When we see one of these followed by ':', we know the previous field has
# ended even if its value spanned multiple visual lines.
_FIELD_RE = re.compile(
    r'^(URL|Refering-URL|Root-URL|Email|Title|Author|Date|Note|Address|'
    r'Affiliation|Abstract|Abstract-found|Intro-found|Reference|'
    r'Reference-contexts|References-found):\s*(.*)$'
)


def parse_extraction_file(path: Path) -> Dict[str, str]:
    """Parse a single MIME-like extraction file.

    Returns a dict of field -> value. Multi-line fields (notably Abstract)
    are concatenated. Multi-occurrence fields (Reference, Reference-contexts)
    are dropped — we only care about Title and Abstract.
    """
    fields: Dict[str, str] = {}
    current_field: Optional[str] = None
    current_buf: List[str] = []

    def flush() -> None:
        nonlocal current_field, current_buf
        if current_field is not None and current_field not in {
            'Reference',
            'Reference-contexts',
        }:
            # First occurrence wins (some fields can repeat for non-Reference)
            if current_field not in fields:
                fields[current_field] = ' '.join(current_buf).strip()
        current_field = None
        current_buf = []

    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.rstrip('\n')
                m = _FIELD_RE.match(line)
                if m:
                    flush()
                    current_field = m.group(1)
                    current_buf = [m.group(2)]
                else:
                    # Continuation of the previous field
                    if current_field is not None:
                        current_buf.append(line.strip())
        flush()
    except FileNotFoundError:
        return {}
    return fields


def build_text_for_paper(
    paper_id: str,
    pid_to_files: Dict[str, List[str]],
    extractions_dir: Path,
) -> Tuple[str, str]:
    """Find the best (title, abstract) for a paper.

    Tries each candidate filename in order, prefers the one with both a
    Title and an Abstract whose Abstract-found is "1". Falls back to title-
    only or empty if nothing is found.

    Returns (combined_text, status) where status is one of:
        "title+abstract", "title_only", "abstract_only", "missing"
    """
    candidates = pid_to_files.get(paper_id, [])
    best_title = ''
    best_abstract = ''
    abstract_was_found = False

    for fname in candidates:
        fpath = extractions_dir / fname
        fields = parse_extraction_file(fpath)
        if not fields:
            continue
        title = fields.get('Title', '').strip()
        abstract = fields.get('Abstract', '').strip()
        ab_found = fields.get('Abstract-found', '').strip() == '1'

        # Prefer files where Abstract-found is 1.
        if ab_found and abstract:
            return _format_text(title, abstract), 'title+abstract'

        # Otherwise track the best-so-far.
        if title and not best_title:
            best_title = title
        if abstract and not best_abstract:
            best_abstract = abstract
            abstract_was_found = ab_found

    if best_title and best_abstract:
        return _format_text(best_title, best_abstract), 'title+abstract'
    if best_title:
        return best_title, 'title_only'
    if best_abstract:
        return best_abstract, 'abstract_only'
    return '', 'missing'


def _format_text(title: str, abstract: str) -> str:
    """Combine title + abstract into one string, mirroring TAPE/OFA style."""
    title = re.sub(r'\s+', ' ', title).strip()
    abstract = re.sub(r'\s+', ' ', abstract).strip()
    if title and abstract:
        return f'{title}. {abstract}'
    return title or abstract


def collect_texts(
    paper_ids: List[str],
    pid_to_files: Dict[str, List[str]],
    extractions_dir: Path,
) -> Tuple[List[str], Dict[str, int]]:
    """Build texts list (one per paper_id) and a status histogram."""
    texts: List[str] = []
    status_counts: dict[str, int] = defaultdict(int)
    for pid in paper_ids:
        text, status = build_text_for_paper(pid, pid_to_files, extractions_dir)
        texts.append(text)
        status_counts[status] += 1
    return texts, dict(status_counts)


# ----------------------------------------------------------------------------
# Splits (UniGraph paper Appendix B)
# ----------------------------------------------------------------------------
# TODO: Check why train < test, val
def make_cora_splits(
    y: torch.Tensor,
    num_nodes: int,
    seed: int = 0,
    train_per_class: int = 20,
    val_per_class: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unigraph paper describes: For Cora and PubMed, we follow commonly used
    data splits, using 20 labeled nodes per class as the training set,
    30 nodes per class as the validation set, and the rest as the test
    set. We report the average accuracy on test set with 20 random
    initialization.
    """
    rng = np.random.default_rng(seed)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    num_classes = int(y.max().item()) + 1
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(idx)
        if len(idx) < train_per_class + val_per_class:
            raise ValueError(
                f'Class {c} has only {len(idx)} samples; need at least '
                f'{train_per_class + val_per_class}.'
            )
        train_mask[idx[:train_per_class]] = True
        val_mask[idx[train_per_class : train_per_class + val_per_class]] = True

    test_mask = ~(train_mask | val_mask)
    return train_mask, val_mask, test_mask


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_file)

    content_path = args.raw_dir / 'cora.content'
    cites_path = args.raw_dir / 'cora.cites'
    papers_path = args.raw_dir / 'mccallum' / 'cora' / 'papers'
    extractions_dir = args.raw_dir / 'mccallum' / 'cora' / 'extractions'
    for p, label in [
        (content_path, 'cora.content'),
        (cites_path, 'cora.cites'),
        (papers_path, 'mccallum/cora/papers'),
        (extractions_dir, 'mccallum/cora/extractions/'),
    ]:
        if not p.exists():
            raise FileNotFoundError(f'Missing {label} at {p}')

    # ---- Graph (cora.content + cora.cites) ----
    logging.info('Parsing cora.content ...')
    paper_ids, y, id_to_idx = parse_cora_content(content_path)
    num_nodes = len(paper_ids)
    logging.info(f'  {num_nodes:,} nodes, {int(y.max().item()) + 1} classes')

    logging.info('Parsing cora.cites ...')
    edge_index = parse_cora_cites(cites_path, id_to_idx)
    logging.info(f'  {edge_index.size(1):,} directed edges (= 2 x undirected)')

    # ---- Splits ----
    train_mask, val_mask, test_mask = make_cora_splits(y, num_nodes, seed=args.seed)
    logging.info(
        f'Splits (seed={args.seed}): '
        f'train={int(train_mask.sum())}, '
        f'val={int(val_mask.sum())}, '
        f'test={int(test_mask.sum())}'
    )

    # ---- Texts (McCallum extractions) ----
    logging.info('Building paper_id -> filename index from McCallum papers ...')
    pid_to_files = parse_mccallum_papers(papers_path)

    # Coverage check: how many of our 2708 papers appear in the index?
    in_index = sum(1 for pid in paper_ids if pid in pid_to_files)
    logging.info(
        f'Cora.content papers found in McCallum index: {in_index}/{num_nodes} '
        f'({in_index / num_nodes:.1%})'
    )
    if in_index < num_nodes:
        missing_examples = [pid for pid in paper_ids if pid not in pid_to_files][:5]
        logging.warning(
            f'  Examples of missing IDs: {missing_examples}. '
            f'These will end up with empty text (placeholder used).'
        )

    logging.info('Reading extraction files (this is the slow part) ...')
    texts, status_counts = collect_texts(paper_ids, pid_to_files, extractions_dir)
    logging.info(f'Text extraction status: {status_counts}')

    # Quick sanity on text quality
    nonempty_lens = [len(t) for t in texts if t]
    if nonempty_lens:
        logging.info(
            f'Text length stats over {len(nonempty_lens)} non-empty texts: '
            f'min={min(nonempty_lens)} median={int(np.median(nonempty_lens))} '
            f'max={max(nonempty_lens)}'
        )

    # ---- PyG Data ----
    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.label_names = CORA_LABEL_NAMES
    data.label_texts = CORA_LABEL_TEXTS
    data.task_type = 'node'
    data.num_classes = len(CORA_LABEL_NAMES)
    torch.save(data, args.out_dir / 'graph.pt')
    logging.info(f'Saved graph to {args.out_dir / "graph.pt"}')

    # ---- Tokenized text store ----
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    write_text_store(
        out_dir=args.out_dir,
        texts=texts,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        task_type='node',
    )


if __name__ == '__main__':
    main()
