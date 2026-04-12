
import pandas as pd
import pytest

from tgfm.processing.deduplication.main import run_deduplication
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
        / "sample_data_duplicate.parquet"
    )
    path_curated = (
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "curated"
        / "deduplicated"
    )
    path_cache = (
        root / "data" / "test_data" / "nemo_curator" / "data" / "curated" / "cache"
    )
    path_cache.mkdir(parents=True, exist_ok=True)
    path_curated.mkdir(parents=True, exist_ok=True)
    ids_to_remove_path = (
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "curated"
        / "ids_to_remove"
    )
    ids_to_remove_path.mkdir(parents=True, exist_ok=True)
    run_deduplication(
        file_paths=file_paths,
        output_path=path_curated,
        ids_to_remove_path=ids_to_remove_path,
        cache_path=path_cache,
        num_cpus=8,
        num_gpus=1,
        deduplication_method="fuzzy",
    )


@pytest.fixture
def read_parquet():
    root = get_root_dir()

    df = pd.read_parquet(
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "sample"
        / "sample_data_duplicate.parquet",
        engine="pyarrow",
    )
    df_cur = pd.read_parquet(
        root
        / "data"
        / "test_data"
        / "nemo_curator"
        / "data"
        / "curated"
        / "deduplicated"
        / "4f71d8c54c86.parquet",
        engine="pyarrow",
    )
    return {"original": df, "curated": df_cur}


def test_difference_parquet_deduplication_minhash(read_parquet):

    df_sample = read_parquet["original"]
    df_deduplicated = read_parquet["curated"]

    print(f"Sample pre-deduplication: {df_sample}")
    print(f"Sample post-deduplication: {df_deduplicated}")

    print(
        f"Pre-deduplication: {len(df_sample)}, Post-deduplication: {len(df_deduplicated)}"
    )

    assert len(df_sample) > len(df_deduplicated)

    removed_domains = df_deduplicated["domain"].values
    assert "duplicated-site-1.com" not in removed_domains
    assert "duplicated-site-2.com" not in removed_domains
    assert "duplicated-site-3.com" not in removed_domains


def test_maintained_source_file_column(read_parquet):
    read_parquet["original"]
    df_deduplicated = read_parquet["curated"]

    columns = [c for c in df_deduplicated.columns]

    assert '__source_file' in columns
