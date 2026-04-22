import logging

import torch
import torch.nn as nn


def compute_mlm_loss(
    mlm_logits: torch.Tensor, input_ids: torch.Tensor, masked_input_ids: torch.Tensor
) -> torch.Tensor:
    mask = input_ids != masked_input_ids
    logging.info(f'mask: {mask}')
    logging.info(f'masked_input_ids values: {masked_input_ids}')

    target_ids = input_ids[mask]
    logging.info(f'target_ids: {target_ids}')

    assert target_ids.shape[0] == mlm_logits.shape[0], (
        f'Shape mismatch: Logits {mlm_logits.shape}, Targets {target_ids.shape}. '
        'Ensure exactly one token is masked per batch item.'
    )

    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(mlm_logits, target_ids)
