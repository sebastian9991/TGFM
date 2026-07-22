"""Zero-shot node classification on target TAGs (GraphCLIP Table 2 protocol).

For each target dataset: random 20% test split, repeated over
``data_args.eval_seeds`` (5 seeds in the paper); mean accuracy +/- std.

Prediction: nearest label sentence by cosine similarity. Embeddings are NOT
l2-normalized during training (SIGReg targets an isotropic Gaussian in R^d);
normalization is applied here at evaluation time only, as in LeVLJEPA
Sec. 2.1. Scoring direction is controlled by ``zeroshot_direction``:
    text_pred  : cos(z_g, h_t(z_t))   — LeVLJEPA's reported direction
    graph_pred : cos(h_g(z_g), z_t)
    direct     : cos(z_g, z_t)

Requires the GraphCLIP repo modules on PYTHONPATH for load_data /
parse_target_data / split_dataloader and the eval prompt templates.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import torch
from args import LeGTJEPAArguments, parse_args
from models import LeGTJEPA
from torch import Tensor
from torch.nn.functional import normalize
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tgfm.dataset.evaluation.load import load_data
from tgfm.utils.args import parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.process import parse_target_data, split_dataloader

EVAL_TEMPLATE = {
    'cora': 'this paper has a topic on {c}',
    'citeseer': 'good paper of {c} ',
    'pubmed': 'it belongs to {c} research area',
    'arxiv_2023': 'it belongs to {c} research area',
    'wikics': 'it belongs to {c} research area',
    'photo': 'this product belongs to {c}',
    'computer': 'is {c} category',
    'history': 'this book belongs to {c}',
    'instagram': '{c}',
    'reddit': '{c}',
}


parser = argparse.ArgumentParser(
    description='Evaluation of LeGTJEPA.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


@torch.no_grad()
def encode_label_sentences(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    classes: List[str],
    c_descs: List[str],
    name: str,
    device: torch.device,
    max_length: int,
) -> Tensor:
    prompts = [
        EVAL_TEMPLATE[name].format(c=c) + desc for c, desc in zip(classes, c_descs)
    ]
    batch_t = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).to(device)
    return model.encode_text(batch_t['input_ids'], batch_t['attention_mask'])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    z_labels: Tensor,
    direction: str,
    device: torch.device,
) -> float:
    model.eval()
    if direction == 'text_pred':
        z_labels = model.text_predictor(z_labels)
    z_labels = normalize(z_labels, dim=-1)

    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        z_g = model.encode_graph(batch)
        if direction == 'graph_pred':
            z_g = model.graph_predictor(z_g)
        z_g = normalize(z_g, dim=-1)
        pred = (z_g @ z_labels.T).argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.numel()
    return correct / total


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    for experiment, experiment_arg in experiment_args.items():
        model_args = experiment_args.model_args
        data_args = experiment_args.data_args
        assert isinstance(model_args, LeGTJEPAArguments)
        setup_logging(meta_args.log_file_path)

        device = torch.device(
            f'cuda:{model_args.device}' if torch.cuda.is_available() else 'cpu'
        )
        model = LeGTJEPA(model_args).to(device)
        ckpt_path = str(
            Path(meta_args.root_dir) / 'weights' / 'LeGTJEPA' / 'legtjepa.pt'
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logging.info('Loaded %s (epoch %d)', ckpt_path, ckpt['epoch'])
        tokenizer = AutoTokenizer.from_pretrained(model_args.text_model_id)

        for name in data_args.target_data.split('+'):
            data, _, classes, c_descs = load_data(name, seed=0)
            target_graphs = parse_target_data(name, data)
            z_labels = encode_label_sentences(
                model,
                tokenizer,
                classes,
                c_descs,
                name,
                device,
                data_args.max_text_length,
            )

            accs = []
            for seed in data_args.eval_seeds:
                seed_everything(seed)
                _, _, test_loader = split_dataloader(
                    data, target_graphs, data_args.batch_size, seed=seed, name=name
                )
                acc = evaluate(
                    model, test_loader, z_labels, model_args.zeroshot_direction, device
                )
                accs.append(acc)
            accs_t = torch.tensor(accs)
            logging.info(
                '%s: %.2f +/- %.2f (direction=%s, seeds=%s)',
                name,
                100 * accs_t.mean(),
                100 * accs_t.std(),
                model_args.zeroshot_direction,
                data_args.eval_seeds,
            )


if __name__ == '__main__':
    main()
