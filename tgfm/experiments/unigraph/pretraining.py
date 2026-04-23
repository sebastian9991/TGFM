import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader
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
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    total_latent_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        logging.info(f'batch size: {batch.batch_size}')
        logging.info(f'Number of ids: {len(batch.n_id)}')
        logging.info(f'batch edge-index: {batch.edge_index.shape}')
        logging.info(f'batch n_id: {batch.n_id}')
        optimizer.zero_grad()

        text_features = text_store.get_features(batch.n_id, apply_masking=True)

        input_ids = text_features['input_ids'].cuda()  # [B, seq]
        masked_input_ids = text_features['masked_input_ids'].cuda()  # [B, seq]
        attention_mask = text_features['attention_mask'].cuda()  # [B, seq]
        token_type_ids = text_features['token_type_ids'].cuda()  # [B, seq]
        edge_index = batch.edge_index.to(device)  # [2, num_edges]

        logging.info(
            f'Shape of input_ids, masked_input_ids, attention_mask, token_type_ids: {input_ids.shape}, {masked_input_ids.shape}, {attention_mask.shape}, {token_type_ids.shape}'
        )

        with autocast():  # Uses FP16 if possible.
            loss, latent_loss = model(
                input_ids,
                masked_input_ids,
                attention_mask,
                token_type_ids,
                edge_index,
                device,
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
    dataset: ClusterData,
    text_store: MAG240MMapTextStore,
    save_dir: Path,
) -> None:
    assert isinstance(model_args, UnigraphArguments)
    data = dataset[0]
    logging.info(f'Type dataset: {type(dataset)}')
    logging.info(f'Type dataset: {type(data)}')

    # loader = NeighborLoader(
    #     data,
    #     input_nodes=torch.arange(data.num_nodes),
    #     num_neighbors=model_args.num_neighbors,
    #     batch_size=model_args.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     prefetch_factor=2,
    #     persistent_workers=True,
    # )

    loader = ClusterLoader(data=dataset, batch_size=1)

    model = UniGraph(model_args).to(model_args.device)

    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_args.lr, weight_decay=model_args.weight_decay
    )

    for epoch in range(model_args.epochs):
        pretrain_loss, pretrain_latent_loss = train_pretrain(
            model,
            loader,
            text_store,
            pretrain_optimizer,
            epoch,
            model_args.device,
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
    cluster_data = ClusterData(
        dataset[0],
        num_parts=dataset[0].num_nodes // 128,
        save_dir=dataset.processed_dir,
    )
    logging.info(f'Clustered Graph Data.')
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
            dataset=cluster_data,
            text_store=text_store,
            save_dir=root_dir / 'weights',
        )


if __name__ == '__main__':
    main()
