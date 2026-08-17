import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import yaml
from hf_argparser import HfArgumentParser

from tgfm.utils.path import get_root_dir, get_scratch

VALID_TASKS = {'graph', 'node', 'link'}


@dataclass
class MetaArguments:
    """Configuration for data locations, file paths, and global experiment setting."""

    log_file_path: Optional[str] = field(
        metadata={'help': 'Path to the log file.'},
    )
    root_dir: Union[str, List[str]] = field(
        metadata={'help': 'Root Directory for meta files.'},
    )
    node_file: Union[str, List[str]] = field(
        metadata={
            'help': 'A csv or list of csv files containing the nodes of the graph.'
        },
    )
    edge_file: Union[str, List[str]] = field(
        metadata={
            'help': 'A csv or list of csv files containing the nodes of the graph.'
        },
    )
    global_seed: int = field(
        default=42,
        metadata={'help': 'Random seed to use for reproducibiility.'},
    )
    is_scratch_location: bool = field(
        default=False,
        metadata={'help': 'Whether to use the /NOBACKUP/ or /SCRATCH/ disk on server.'},
    )
    verbose: bool = field(
        default=False,
        metadata={'help': 'Whether to track metrics with wandb.'},
    )
    wandb_offline: bool = field(
        default=True, metadata={'help': 'Whether wandb is offline.'}
    )
    wandb_sweep_id: str = field(
        metadata={'help': 'The sweep id for evaluation on wandb sweep weights.'}
    )

    def __post_init__(self) -> None:
        """Resolve all file and directory paths relative to the selected root directory."""
        # Select root directory
        # TODO: We now assume absolute path is given in configuration.
        root_dir = get_scratch() if self.is_scratch_location else get_root_dir()

        def resolve_paths(files: Union[str, List[str]]) -> Union[str, List[str]]:
            def resolve(f: str) -> str:
                return str(
                    root_dir / f.lstrip('/')
                )  # TODO: This breaks with a List[str]?

            if isinstance(files, str):
                return resolve(files)
            return [resolve(f) for f in files]

        self.node_file = resolve_paths(self.node_file)
        self.edge_file = resolve_paths(self.edge_file)
        self.root_dir = resolve_paths(self.root_dir)

        if self.log_file_path is not None:
            log_dir = get_root_dir() / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = str(log_dir / self.log_file_path)


@dataclass
class DataArguments:
    """Configuration of task-level data and problem settings."""

    data_name: str = field(
        metadata={'help': 'The name of the dataset, needed for pre-built datasets.'},
    )
    task_name: str = field(
        metadata={'help': 'The name of the task to train on'},
    )
    pretrain_data_size: int = field(
        metadata={'help': 'Pretraining data size.'}, default=-1
    )
    target_data: str = field(
        default='',
        metadata={
            'help': 'The datasets used in target evaluation concatenated by +. As per graphCLIP.'
        },
    )
    eval_data_names: List[str] = field(
        default_factory=list,
        metadata={
            'nargs': '+',
            'metavar': 'n',
            'help': 'One or more eval datasets from PyG natives.',
        },
    )
    pretrain_data_ids: List[int] = field(
        default_factory=lambda: [-1],
        metadata={
            'nargs': '+',
            'metavar': 'n',
            'help': 'One or more eval datasets from PyG natives.',
        },
    )
    num_test_shards: int = field(
        metadata={'help': 'Number of test splits to do for uncertainty estimates.'},
        default=1,
    )
    transform: bool = field(
        default=False,
        metadata={'help': 'Whether to transform dataset.'},
    )
    test_ratio: float = field(default=0.2)
    eval_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    eval_seeds_sweep: int = field(default=2)
    eval_batch_size: int = field(default=256)
    eval_every_epochs: int = field(default=5)

    def __post_init__(self) -> None:
        if self.task_name not in VALID_TASKS:
            raise ValueError(
                f"Invalid task name '{self.task_name}'. Must be one of {VALID_TASKS}"
            )


@dataclass
class ModelArguments:
    """Configuration of model architecture and training hyperparameters."""

    model: str = field(
        metadata={'help': 'Unique model identifer and registry.'},
    )
    lr: float = field(default=0.001, metadata={'help': 'Learning Rate.'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout value.'})
    weight_decay: float = field(
        default=2.36e-5, metadata={'help': 'Weight decay on the optimizer.'}
    )
    batch_size: int = field(default=32)
    num_steps: int = field(
        default=100, metadata={'help': 'The number of steps of training.'}
    )
    epochs: int = field(default=100, metadata={'help': 'Number of epochs.'})
    runs: int = field(default=3, metadata={'help': 'Number of trials.'})
    patience: int = field(
        default=10,
        metadata={'help': 'Number of epochs to wait before no validation improvement.'},
    )
    eval_frequency: int = field(
        default=500, metadata={'help': 'The frequency of evaluation.'}
    )
    eval_repeat: int = field(default=1, metadata={'help': 'The repeats of evaluation.'})
    log_frequency: int = field(
        default=10, metadata={'help': 'The frequency of logging.'}
    )
    edge_drop_rate: float = field(default=0.3)  # p_d
    feat_mask_rate: float = field(default=0.1)  # p_m
    use_cuda: bool = field(default=True, metadata={'help': 'Whether to use cuda.'})
    device: int = field(default=0, metadata={'help': 'Device to be used.'})


@dataclass
class UnigraphArguments(ModelArguments):
    """Configuration of model architecture and training hyperparameters."""

    model: str = 'Unigraph'
    lm_type: str = field(default='microsoft/deberta-base')
    hidden_size: int = field(default=768)
    num_neighbors: list[int] = field(
        default_factory=lambda: [-1],
    )
    num_layers: int = field(default=3)
    nhead: int = field(default=8)
    activation: str = field(default='gelu')
    norm: str = field(default='layernorm')
    gradient_checkpointing: bool = field(default=False)
    negative_slope: float = field(default=0.2)
    mask_rate: float = field(default=0.15)
    lam: float = field(default=0.1)
    momentum: float = field(default=0.996)
    delayed_ema_epoch: int = field(default=10)


@dataclass
class SimpleMPNN(ModelArguments):
    """Configuration of model architecture and training hyperparameters."""

    model: str = 'SimpleMPNN'
    # pretraining
    lam: float = field(default=0.1)  # uniformity weight
    edge_drop_rate: float = field(default=0.3)  # p_d
    feat_mask_rate: float = field(default=0.1)  # p_m
    hid_dims: list = field(default_factory=lambda: [256, 256])
    encoder: str = field(default='gcn')  # 'gcn' or 'mlp' (CoauthorCS)

    rwse_K: int = field(default=16)  # K for RWSE

    num_global_views: int = field(default=2)
    num_local_parts: int = field(default=8)
    global_coverage_frac: float = field(default=0.7)
    global_strategy: str = field(default='bfs')
    num_local_as_global: int = field(default=0)

    lambd: float = field(default=0.05)
    num_slices: int = field(default=256)

    # probe (paper eval)
    lr2: float = field(default=1e-2)
    wd2: float = field(default=1e-4)


@dataclass
class GraphGPSArguments(ModelArguments):
    """Configuration of model architecture and training hyperparameters."""

    model: str = 'GraphGPS'
    dim: int = field(default=128)
    num_layers: int = field(default=3)
    num_heads: int = field(default=4)
    local_gnn_type: str = field(default='GINE')
    attn_type: str = field(
        default='multihead'
    )  # PyG GPSConv: "multihead" | "performer"
    norm: str = field(default='batch_norm')  # PyG GPSConv internal norm

    # Will sum to dim, asserted in graphGPS
    # Note: A good ratio is to have dim = node_out_dim + pe_out_dim/3.
    node_out_dim: int = field(default=96)
    pe_out_dim: int = field(default=32)
    se_out_dim: int = field(default=0)

    num_neighbors: list[int] = field(
        default_factory=lambda: [-1],
    )

    rwse_K: int = field(default=16)  # K for RWSE

    num_global_views: int = field(default=2)
    num_local_parts: int = field(default=8)
    global_coverage_frac: float = field(default=0.7)
    global_strategy: str = field(default='bfs')
    num_local_as_global: int = field(default=0)

    lambd: float = field(default=0.05)
    num_slices: int = field(default=256)


@dataclass
class SSGEArguments(ModelArguments):
    """SSGE Configuration of model architecture and training hyperparameters."""

    model: str = 'SSGE'
    # pretraining
    lam: float = field(default=0.1)  # uniformity weight
    hid_dims: list = field(default_factory=lambda: [256, 256])
    encoder: str = field(default='gcn')  # 'gcn' or 'mlp' (CoauthorCS)
    num_slices: int = field(default=256)

    # probe (paper eval)
    lr2: float = field(default=1e-2)
    wd2: float = field(default=1e-4)


@dataclass
class TransferArguments(ModelArguments):
    """Arguments for transfer experiments."""

    # model parameters
    model: str = 'Transfer'
    encoder: str = field(default='GCN')
    hidden_dim: int = field(default=384)
    n_head: int = field(default=4)
    n_layers: int = field(default=2)
    attn_drop: float = field(default=0.1)
    opt: str = field(default='adamw')
    norm: str = field(default='none')
    activation: str = field(default='relu')
    use_residual: bool = field(default=False)

    # training parameters
    scheduler: str = field(default='cosine')
    make_undirected: bool = field(default=False)
    epochs: int = field(default=5)
    warmup_steps: int = field(default=1)
    peak_lr: float = field(default=0.001)
    weight_decay: float = field(default=0.00001)
    dropout: float = field(default=0.1)
    eval_step: int = field(default=1)
    save_step: int = field(default=1)
    num_workers: int = field(default=4)

    # ssl parameters
    task: str = field(default='sigreg')
    p_edge_drop: float = field(default=0.2)
    p_feat_drop: float = field(default=0.2)

    # BGRL parameters
    bgrl_mm: float = field(default=0.99)
    bgrl_pred_hid: int = field(default=512)

    # sigreg parameters
    n_directions: int = field(default=256)
    n_points: int = field(default=17)
    lambda_sigreg: float = field(default=1.0)

    # eval parameters
    n_tasks: int = field(default=5)
    n_shots: int = field(default=5)
    n_val: int = field(default=500)
    eval_data_seed: int = field(default=0)

    # linear probing parameters
    linear_runs: int = field(default=1)
    linear_lr: float = field(default=1e-2)
    linear_l2: float = field(default=1e-4)
    linear_dropout: float = field(default=0.1)

    # LeGTJEPA cross-modal parameters (task: 'legtjepa')
    text_input_dim: int = field(
        default=384, metadata={'help': 'Dim of the precomputed node text embeddings.'}
    )
    embed_dim: int = field(
        default=384, metadata={'help': 'Shared space where SIGReg is computed.'}
    )
    proj_hidden_dim: int = field(default=2048)
    proj_dropout: float = field(default=0.1)
    predictor_depth: int = field(default=4)
    predictor_hidden_dim: int = field(default=2048)
    predictor_dropout: float = field(default=0.1)
    freeze_text_projection: bool = field(
        default=False, metadata={'help': 'Ablation: fully locked text tower.'}
    )
    lambda_graph: float = field(default=0.01)
    lambda_text: float = field(default=0.01)
    sigreg_num_slices: int = field(default=256)
    sigreg_num_quad_points: int = field(default=17)
    sigreg_t_max: float = field(default=4.0)


@dataclass
class LeGTJEPAArguments(ModelArguments):
    model: str = 'LeGTJEPA'

    # TODO: Make sure the source data is all of what is provided by GraphCLIP repo
    # --- source data (GraphCLIP layout under data_root) ---
    source_data: str = 'pubmed'
    batch_size: int = 900  # per rank; global = batch_size * world_size
    max_text_length: int = 512

    log_every_steps: int = 100

    # --- graph encoder (GraphGPS, matched to GraphCLIP Base) ---
    graph_in_dim: int = 384  # SBERT node features
    graph_hidden_dim: int = 1024
    graph_num_layers: int = 12
    graph_pe_dim: int = 8
    attn_type: str = 'multihead'
    attn_dropout: float = 0.0

    # --- text encoder ---
    text_model_id: str = 'sentence-transformers/all-MiniLM-L6-v2'
    freeze_text_backbone: bool = True  # GraphCLIP-style locked text tower
    freeze_text_projection: bool = False  # ablation: fully frozen text side

    # --- shared embedding space / projections (LeVLJEPA Sec. 3.2) ---
    embed_dim: int = 384
    proj_hidden_dim: int = 2048
    proj_dropout: float = 0.1

    # --- cross-modal predictors (LeVLJEPA App. A: depth 4, width 2048) ---
    predictor_depth: int = 4
    predictor_hidden_dim: int = 2048
    predictor_dropout: float = 0.1

    # --- objective ---
    lambda_graph: float = 0.01
    lambda_text: float = 0.01
    sigreg_num_slices: int = 256
    sigreg_num_quad_points: int = 17
    sigreg_t_max: float = 4.0

    # --- optimization ---
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    epochs: int = 30
    warmup_steps: int = 1000

    # TODO: Check this
    # --- evaluation ---
    # 'text_pred'  : cos(z_g, h_t(z_t))   (LeVLJEPA's reported direction)
    # 'graph_pred' : cos(h_g(z_g), z_t)
    # 'direct'     : cos(z_g, z_t)
    # 'text_pred+graph_pred' : cos(h_g(z_g), h_t(z_t))
    zeroshot_direction: str = 'text_pred'

    # --- GraphCLIP-style stochasticity
    graph_aug: bool = False
    aug_feat_drop: float = 0.3  # GraphCLIP: graph_aug(g, 0.3, 0.2)
    aug_edge_drop: float = 0.2
    adversarial: bool = False
    adv_step_size: float = 1.0e-2  # GraphCLIP: step_size=1e-2, m=3
    adv_steps: int = 3


@dataclass
class ExperimentArgument:
    """Container for a single experiment's data and model configuration."""

    data_args: DataArguments = field(
        metadata={'help': 'Data arguments for GNN configuration.'}
    )
    model_args: ModelArguments = field(
        metadata={'help': 'Model arguments for the GNN.'}
    )


@dataclass
class ExperimentArguments:
    """Collection of named experiments and their configurations."""

    exp_args: Dict[str, ExperimentArgument] = field(
        metadata={'help': 'List of experiments.'}
    )

    def __post_init__(self) -> None:
        """Convert experiment dictionaries into ExperimentArgument instances."""

        def _remap_experiment_args(
            experiments: Dict[str, ExperimentArgument],
        ) -> Dict[str, ExperimentArgument]:
            for exp_name, exp_val in experiments.items():
                if isinstance(exp_val, dict):
                    model_args = get_model_args(exp_val['model_args'])
                    data_args = DataArguments(**exp_val['data_args'])
                    experiments[exp_name] = ExperimentArgument(
                        model_args=model_args,
                        data_args=data_args,
                    )
            return experiments

        self.exp_args = _remap_experiment_args(self.exp_args)


MODEL_REGISTRY: Dict[str, Type[ModelArguments]] = {
    'Unigraph': UnigraphArguments,
    'GraphGPS': GraphGPSArguments,
    'SSGE': SSGEArguments,
    'SimpleMPNN': SimpleMPNN,
    'Transfer': TransferArguments,
    'LeGTJEPA': LeGTJEPAArguments,
}


def get_model_args(config: Dict[str, Any]) -> ModelArguments:
    """Factory to instantiate the correct model class."""
    model_type = config.get('model')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f'Unknown model type: {model_type}. Must be one of {list(MODEL_REGISTRY.keys())}'
        )

    # Instantiate the specific dataclass
    return MODEL_REGISTRY[model_type](**config)


def parse_args(
    config_yaml: Union[str, pathlib.Path],
) -> Tuple[MetaArguments, ExperimentArguments]:
    """Parse a YAML configuration file into typed argument objects.

    Parameters:
        config_yaml : Union[str, pathlib.Path]
            Path to the YAML configuration file.

    Returns:
        Tuple[MetaArguments, ExperimentArguments]
            Parsed meta and experiment configuration objects.
    """
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())
    config_dict = config_dict['MetaArguments'] | config_dict['ExperimentArguments']
    parser = HfArgumentParser((MetaArguments, ExperimentArguments))
    return parser.parse_dict(config_dict, allow_extra_keys=True)
