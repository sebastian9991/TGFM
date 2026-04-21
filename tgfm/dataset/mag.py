import logging
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset

from tgfm.dataset.ogb_mag import MAG240MDataset


class MAG240MGraphDataset(InMemoryDataset):
    """InMemory PyTorch Geometric Dataset for OGB MAG240M.

    This dataset loads a single Data object containing:
    - edge_index: Citation graph edges [2, num_edges]
    - texts: List of all node texts (title + abstract) indexed by node_id

    Args:
        root (str): Root directory where the dataset should be saved.
        text_csv_path (str, optional): Path to text.csv from mag240m_mapping.zip.
        transform (callable, optional): A function/transform that takes in a
            Data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in
            a Data object and returns a transformed version.
        pre_filter (callable, optional): A function that takes in a Data object
            and returns a boolean value.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self._mag_dataset: Optional[MAG240MDataset] = None

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the processed data into memory
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def mag_dataset(self) -> MAG240MDataset:
        """Lazy load the MAG240M dataset."""
        if self._mag_dataset is None:
            logging.info(f'Looking at {self.root} for unprocessed MAG Dataset.')
            self._mag_dataset = MAG240MDataset(root=self.root)
        return self._mag_dataset

    @property
    def raw_file_names(self) -> List[str]:
        """Returns a list of raw file names that need to be found in the raw_dir.
        The MAG240M dataset handles its own file structure.
        """
        return []  # MAG240MDataset handles its own files

    @property
    def processed_file_names(self) -> List[str]:
        """Returns a list of processed file names.
        If these files exist in processed_dir, processing is skipped.
        """
        return ['data.pt']

    def download(self) -> None:
        """Download the MAG240M dataset.
        The MAG240MDataset class handles downloading automatically.
        """
        # The MAG240MDataset will download when first accessed
        _ = self.mag_dataset
        logging.info('MAG240M dataset downloaded/verified.')

    def process(self) -> None:
        """Process the raw data and save a single Data object to processed_dir.

        Creates a Data object with:
        - edge_index: Citation edges
        - texts: List of all node texts (indexed by node_id)
        - num_nodes: Total number of nodes
        """
        logging.info('Processing MAG240M dataset...')

        # Load edge index
        logging.info('Loading edge index...')
        edge_index = self.mag_dataset.edge_index('paper', 'cites', 'paper')
        edge_index_tensor = torch.from_numpy(edge_index)
        logging.info(f'Shape of edge_index: {edge_index_tensor.shape}')
        log_edge_index_range(edge_index_tensor)

        num_nodes = self.mag_dataset.num_papers
        logging.info(f'{num_nodes:,} nodes...')

        # Create Data object
        data = Data(
            edge_index=edge_index_tensor,
            num_nodes=num_nodes,
        )

        data_list = [data] if data is not None else []

        logging.info(f'Saving processed data to {self.processed_paths[0]}...')
        self.save(data_list, self.processed_paths[0])
        logging.info('Processing complete.')

    def get_node_text(self, node_id: int) -> str:
        """Get the text for a specific node (title + abstract concatenated).

        Args:
            node_id (int): The node ID.

        Returns:
            str: Concatenated title and abstract.
        """
        return self.data.texts[node_id]

    def get_texts(self, node_ids: List[int]) -> List[str]:
        """Get texts for multiple nodes efficiently.

        Args:
            node_ids (List[int]): List of node IDs.

        Returns:
            List[str]: List of concatenated title + abstract strings.
        """
        return [self.get_node_text(node_id) for node_id in node_ids]

    @property
    def edge_index(self) -> torch.Tensor:
        """Get the full edge index for the graph.
        Shape: [2, num_edges].

        Returns:
            torch.Tensor: Edge index tensor.
        """
        return self.data.edge_index

    @property
    def num_nodes(self) -> int:
        """Get the total number of nodes in the graph."""
        return self.data.num_nodes

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_nodes={self.num_nodes}, num_edges={self.edge_index.shape[1]})'


def log_edge_index_range(edge_index: torch.Tensor) -> None:
    if edge_index.numel() == 0:
        logging.warning('Edge index is empty, cannot compute range.')

    min_indices, _ = edge_index.min(dim=1)
    max_indices, _ = edge_index.max(dim=1)

    logging.info(
        f'Source indices range: [{min_indices[0].item()}, {max_indices[0].item()}]'
    )
    logging.info(
        f'Target indices range: [{min_indices[1].item()}, {max_indices[1].item()}]'
    )
