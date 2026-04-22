import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from tgfm.dataset.mag import MAG240MGraphDataset
from tgfm.dataset.mag_memmap import MAG240MMapTextStore
from tgfm.models.unigraph import UniGraph
from tgfm.utils.args import ModelArguments, UnigraphArguments, parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Pre-training UniGraph on ogb-100M.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


def train_pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    text_store: MAG240MMapTextStore,
    optimizer: torch.optim.AdamW,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    total_latent_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        logging.info(f'batch size: {batch.shape}')
        optimizer.zero_grad()

        text_features = text_store.get_features(batch.n_id, apply_masking=True)

        input_ids = text_features['input_ids'].cuda()
        masked_input_ids = text_features['masked_input_ids'].cuda()
        attention_mask = text_features['attention_mask'].cuda()
        token_type_ids = text_features['token_type_ids'].cuda()

        loss, latent_loss = model(
            input_ids,
            masked_input_ids,
            attention_mask,
            token_type_ids,
            batch.edge_index,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_latent_loss += latent_loss.item()

        pbar.set_postfix(
            {'loss': f'{loss.item():.4f}', 'latent_loss': f'{latent_loss.item():.4f}'}
        )

    return total_loss / len(train_loader), total_latent_loss / len(train_loader)


def run_unigraph(
    model_args: ModelArguments,
    dataset: InMemoryDataset,
    text_store: MAG240MMapTextStore,
    save_dir: Path,
) -> None:
    assert isinstance(model_args, UnigraphArguments)
    data = dataset[0]
    logging.info(f'Type dataset: {type(dataset)}')
    logging.info(f'Type dataset: {type(data)}')

    loader = NeighborLoader(
        data,
        num_neighbors=model_args.num_neighbors,
        batch_size=model_args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    model = UniGraph(model_args).to(model_args.device)

    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay
    )

    for epoch in range(model_args.epochs):
        pretrain_loss, pretrain_latent_loss = train_pretrain(
            model, loader, text_store, pretrain_optimizer, epoch
        )

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': pretrain_optimizer.state_dict(),
                    'pretrain_loss': pretrain_loss,
                    'pretrain_latent_loss': pretrain_latent_loss,
                },
                save_dir / f'pretrain_epoch_{epoch + 1}.pt',
            )


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    seed_everything(meta_args.global_seed)
    setup_logging(meta_args.log_file_path)
    root_dir = Path(str(meta_args.root_dir))  # TODO: Throw when its a list[str]
    dataset = MAG240MGraphDataset(root=str(root_dir))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    logging.info('Tokenizer loaded.')
    text_store = MAG240MMapTextStore(
        output_dir=str(root_dir / 'mag240m_mapping'),
        tokenizer=tokenizer,
    )
    logging.info('Text store loaded.')

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n***Running*** {experiment}')
        run_unigraph(
            model_args=experiment_arg.model_args,
            dataset=dataset,
            text_store=text_store,
            save_dir=root_dir / 'weights',
        )


if __name__ == '__main__':
    main()
