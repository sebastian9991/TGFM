import pandas as pd
import pytest

from tgfm.processing.language_extraction.main import run_language_extraction
from tgfm.utils.path import get_root_dir


@pytest.fixture(scope="session", autouse=True)
def get_deduplication():
    """Verify deduplication on sample dataset."""
    root = get_root_dir()
    file_paths = (
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "sample"
        / "multiple"
        / "languages"
    )
    output_path = (
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "curated"
        / "multiple"
        / "languages"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    fast_text_path = root / "fast_text"
    run_language_extraction(
        file_paths=file_paths,
        output_path=output_path,
        fast_text_path=fast_text_path,
        files_per_partition=1,
        num_cpus=8,
        num_gpus=1,
    )


@pytest.fixture
def read_parquet():
    root = get_root_dir()
    original_path = (
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "sample"
        / "multiple"
        / "languages"
    )

    curated_path = (
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "curated"
        / "multiple"
        / "languages"
    )

    samples = []
    for path in original_path.glob("*.parquet"):
        print(f"Paths: {path}")
        samples.append(pd.read_parquet(path, engine="pyarrow", dtype_backend="pyarrow"))

    df = pd.concat(samples)

    curated_samples = []
    for path in curated_path.glob("*.parquet"):
        print(f"Paths: {path}")
        curated_samples.append(
            pd.read_parquet(path, engine="pyarrow", dtype_backend="pyarrow")
        )

    df_cur = pd.concat(curated_samples)
    return {"original": df, "curated": df_cur}


def test_maintained_size(read_parquet):
    original = read_parquet["original"]
    curated = read_parquet["curated"]

    assert len(original) == len(curated)


def test_maintained_source_file_column(read_parquet):
    read_parquet["original"]
    df_curated = read_parquet["curated"]

    columns = [c for c in df_curated.columns]

    assert "__source_file" in columns
