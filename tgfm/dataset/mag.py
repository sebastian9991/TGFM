import logging
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from tgfm.dataset.ogb_mag import MAG240MDataset


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
        text_csv_path: Optional[str] = None,
        idx_to_paperid_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.text_csv_path = text_csv_path
        self.idx_to_paperid_path = idx_to_paperid_path
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

    def _get_text_csv_path(self) -> Tuple[str, str]:
        """Get the path to text.csv, downloading if necessary."""
        if self.text_csv_path is not None and self.idx_to_paperid_path is not None:
            return (self.text_csv_path, self.idx_to_paperid_path)

        # Default path
        mapping_dir = os.path.join(self.root, 'mag240m_mapping')
        csv_path = os.path.join(mapping_dir, 'text.csv')
        idx_to_paperid_path = os.path.join(mapping_dir, 'paperidx2paperid.csv')

        if os.path.exists(csv_path) and os.path.exists(idx_to_paperid_path):
            return (csv_path, idx_to_paperid_path)
        else:
            raise FileNotFoundError(
                f'One of paths: {csv_path}, {idx_to_paperid_path} has no text.csv or paperidx2paperid.csv file.'
            )

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

        # Load text CSV
        csv_path, idx_to_paperid_path = self._get_text_csv_path()
        logging.info(f'Loading text from {csv_path}...')
        text_df = pd.read_csv(
            csv_path,
            keep_default_na=False,
            na_values=[''],
        )
        logging.info(f'Loading idx_to_paperid from {idx_to_paperid_path}...')
        pd.read_csv(idx_to_paperid_path)
        text_df['title'] = text_df['title'].fillna('')
        text_df['abstract'] = text_df['abstract'].fillna('')

        # Get number of nodes
        num_nodes = self.mag_dataset.num_papers
        logging.info(f'Processing {num_nodes:,} nodes...')

        # Create texts list indexed by node_id
        # Initialize with empty strings
        texts = [''] * num_nodes

        # Fill in texts from CSV
        for _, row in tqdm(text_df.iterrows(), desc='Processing text'):
            idx = int(row['idx'])
            title = row['title']
            abstract = row['abstract']

            # Concatenate with a separator
            if title and abstract:
                text = f'{title}. {abstract}'
            elif title:
                text = title
            elif abstract:
                text = abstract
            else:
                text = ''

            # Add [CLS] and [SEP] tokens
            text = f'[CLS] {text} [SEP]'

            texts[idx] = text

        logging.info(f'Processed {len([t for t in texts if t])} non-empty texts')

        # Create Data object
        data = Data(
            edge_index=edge_index_tensor,
            texts=texts,
            num_nodes=num_nodes,
        )

        data_list = [data] if data is not None else []

        # Use collate to create slices
        data, slices = self.collate(data_list)

        logging.info(f'Saving processed data to {self.processed_paths[0]}...')
        self.save((data, slices), self.processed_paths[0])
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


# Example usage
if __name__ == '__main__':
    # Initialize dataset
    dataset = MAG240MGraphDataset(root='./data/mag240m')

    print(f'Dataset: {dataset}')
    print(f'Number of nodes: {dataset.num_nodes:,}')
    print(f'Number of edges: {dataset.edge_index.shape[1]:,}')

    # Access edge index (for use with NeighborLoader)
    edge_index = dataset.edge_index
    print(f'\nEdge index shape: {edge_index.shape}')

    # Access node text
    text = dataset.get_node_text(0)
    print(f'\nNode 0 text: {text[:200]}...')

    # Get multiple texts efficiently
    texts = dataset.get_texts([0, 1, 2, 3, 4])
    for i, t in enumerate(texts):
        print(f'Node {i}: {t[:100]}...')
