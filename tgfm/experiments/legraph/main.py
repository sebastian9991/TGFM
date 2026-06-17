"""Training loop for LeGraph.

Key design points:
    - Single shared encoder (no EMA, no teacher-student, no predictor).
    - Per-step: prep views for B subgraphs, encode all views in one batched
      forward, compute predictive and SIGReg loss, backprop, step.
    - Logging: every `log_freq` steps, run encoder over a held-out probe batch
      and logging collapse diagnostics.

View construction & encoding utilities.

These mirror the pseudo-code:
    - embed_subgraph_region(...)  -> z (d,)
    - build_views(...)            -> (z_global, z_local)

The encoder is *shared* across all views (no teacher-student).

Each view in the input batch is represented as a PyG-style `Data` (or a dict)
with the fields:
    x:          (|V|, in_dim)         node features for this view
    edge_index: (2, |E|)               edges within the view
    pe:         (|V|, K)               positional encoding (e.g. RWSE) restricted to view nodes
    edge_attr:  (|E|, ?)               optional, edge features

To run efficiently we batch all views of all subgraphs in one `Batch`. We then
mean-pool by `batch` to get one (d,) embedding per view, and reshape back to
(B, V_g, d) and (B, V_l, d) via the bookkeeping returned from the dataloader.
"""

import argparse
from pathlib import Path

from tgfm.experiments.legraph.trainers import train_graph_task, train_node_task
from tgfm.utils.args import parse_args
from tgfm.utils.logger import setup_logging
from tgfm.utils.path import get_root_dir
from tgfm.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Distributed step-based pretraining UniGraph.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file', type=str, required=True, help='Path to configuration file.'
)


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    root_dir = Path(str(meta_args.root_dir))
    seed_everything(meta_args.global_seed)
    setup_logging(meta_args.log_file_path)
    for experiment, experiment_arg in experiment_args.exp_args.items():
        if experiment_arg.data_args.task_name == 'node':
            train_node_task(
                model_args=experiment_arg.model_args,
                data_args=experiment_arg.data_args,
                meta_args=meta_args,
                save_dir=root_dir / 'weights',
            )
        elif experiment_arg.data_args.task_name == 'graph':
            train_graph_task(
                model_args=experiment_arg.model_args,
                data_args=experiment_arg.data_args,
                meta_args=meta_args,
                save_dir=root_dir / 'weights',
            )
        else:
            # TODO: Include link tasks and Runtime Error.
            raise RuntimeError(f'No valid experiment.')


if __name__ == '__main__':
    main()
