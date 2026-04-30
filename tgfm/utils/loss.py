import logging

import torch
import torch.nn as nn


def compute_mlm_loss(
    mlm_logits: torch.Tensor, input_ids: torch.Tensor, masked_input_ids: torch.Tensor
) -> torch.Tensor:
    """Compute the MLM loss on masked positions.
    mlm_logits: [batch_size, seq_len, vocab_size]
    input_ids: [batch_size, seq_len]
    masked_input_ids: [batch_size, seq_len].
    """
    mask = input_ids != masked_input_ids
    logging.info(f'mask: {mask.shape}')
    logging.info(f'masked_input_ids values: {masked_input_ids.shape}')

    labels = input_ids.clone()
    labels[input_ids == masked_input_ids] = -100  # unmasked positions ignored

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    loss = loss_fct(
        mlm_logits.view(-1, mlm_logits.size(-1)),  # [batch_size*seq_len, vocab_size]
        labels.view(-1).long(),  # [batch_size*seq_len]
    )

    return loss
