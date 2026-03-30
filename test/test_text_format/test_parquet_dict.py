import numpy as np
import pandas as pd
import pytest

from tgfm.processing.process_parquet import convert_to_datatrove_format
from tgfm.utils.path import get_root_dir


@pytest.fixture
def read_parquet():
    root = get_root_dir()

    df = pd.read_parquet(root / "data" / "content_test.parquet", engine="pyarrow")
    return df


def test_parquet_format(read_parquet):
    """Verify the parquet format for text extracted via CrediBench."""

    print(read_parquet.head())

    assert len(read_parquet["pages"][0]) > 0

    assert isinstance(read_parquet["pages"][0][0], dict)
    assert "wet_record_txt" in read_parquet["pages"][0][0]



def test_types(read_parquet):
    """Verify the types associated with the parquet pages and domains."""

    assert isinstance(read_parquet["pages"], pd.Series)

    assert isinstance(read_parquet["pages"][0], np.ndarray)

    assert isinstance(read_parquet["pages"][0][0], dict)

    assert isinstance(read_parquet['domain'], pd.Series)



def test_subdomain_occurences(read_parquet):
    """Verify if there are domains with lenghts of lists greater than 1."""
    subdomain_occurences = 0
    subdomain_occured = False
    for idx, row in read_parquet.iterrows():
        if len(row["pages"]) > 1:
            subdomain_occurences += 1
            subdomain_occured = True

    print(f"Number of subdomain occurences: {subdomain_occurences}")
    assert subdomain_occured == True


def test_unique_domain_identifier(read_parquet):
    """Verify if there are domains with lenghts of lists greater than 1."""

    domain_names = set()
    all_unique = True
    for _, row in read_parquet.iterrows():
        if row["domain"] in domain_names:
            all_unique = False
            break
        else:
            domain_names.add(row["domain"])

    assert all_unique == True


def test_convert_to_datatrove_parquet():
    """Verify format of datatrove parquet conversion."""
    root = get_root_dir()
    path = root / "data" / "content_test.parquet"

    df = convert_to_datatrove_format(path)

    print(df.head())

    columns = [c for c in df.columns]

    assert columns == ['domain', 'wet_record_txt', 'WARC_Target_URIs']
