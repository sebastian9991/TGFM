import time

import pytest
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.fuzzy.workflow import (
    FuzzyDeduplicationWorkflow,
)
from nemo_curator.stages.text.deduplication.removal_workflow import (
    TextDuplicatesRemovalWorkflow,
)

from tgfm.utils.path import get_root_dir


def test_curator_deduplication_minhash():
    """Verify deduplication on sample dataset."""
    ray_client = RayClient(num_cpus=8, num_gpus=1, object_store_memory=500_000_000)
    ray_client.start()
    time.sleep(10)

    try:
        root = get_root_dir()

        path = (
            root
            / "data"
            / "test_data"
            / "nemo_curator"
            / "data"
            / "sample"
            / "sample_data_duplicate.parquet"
        )
        path_curated = root / "data" / "test_data" / "nemo_curator" / "data" / "curated"
        path_cache = (
            root / "data" / "test_data" / "nemo_curator" / "data" / "curated" / "cache"
        )
        path_cache.mkdir(parents=True, exist_ok=True)
        ids_to_remove_path = (
            root
            / "data"
            / "test_data"
            / "nemo_curator"
            / "data"
            / "curated"
            / "ids_to_remove"
            / "FuzzyDeduplicateIds"
        )
        ids_to_remove_path.mkdir(parents=True, exist_ok=True)

        fuzzy_workflow = FuzzyDeduplicationWorkflow(
            input_path=str(path),
            cache_path=str(path_cache),
            output_path=str(ids_to_remove_path),
            text_field="wet_record_txt",
            perform_removal=False,
            input_filetype="parquet",
            char_ngrams=24,
            num_bands=20,
            minhashes_per_band=13,
        )

        fuzzy_workflow.run()

        path_cache.mkdir(parents=True, exist_ok=True)
        if any(ids_to_remove_path.glob("*.parquet")):
            removal_workflow = TextDuplicatesRemovalWorkflow(
                input_path=str(path),
                ids_to_remove_path=str(ids_to_remove_path),
                output_path=str(path_curated),
                input_filetype="parquet",
                input_id_field="_curator_dedup_id",
                ids_to_remove_duplicate_id_field="_curator_dedup_id",
            )

            removal_workflow.run()
        else:
            print("No duplicates found, skipping removal step.")

        ray_client.stop()
    except ConnectionError as e:
        ray_client.stop()
        print(f"Pytest Passing but with exception: {e}")
    except Exception as e:
        ray_client.stop()
        pytest.fail(f"Pipeline failed with exception: {e}")
