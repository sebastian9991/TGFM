from pathlib import Path

import pandas as pd


def write_to_parquet_format(
    dataframe: pd.DataFrame, save_dir: Path, file_name: str
) -> None:
    """Write dataframe to parquet format under a specified file location."""
    file_path = save_dir / file_name

    dataframe.to_parquet(file_path, index=False)
