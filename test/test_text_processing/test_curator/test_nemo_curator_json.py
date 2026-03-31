from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import (
    NonAlphaNumericFilter,
    WordCountFilter,
)
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules.score_filter import ScoreFilter

from tgfm.utils.path import get_root_dir


def test_curator_jsonl():
    ray_client = RayClient(num_cpus=8, num_gpus=0, object_store_memory=500_000_000)
    ray_client.start()

    try:
        root = get_root_dir()
        file_paths = str(
            root / "data" / "test_data" / "nemo_curator" / "data" / "sample"
        )
        file_paths_curated = str(
            root / "data" / "test_data" / "nemo_curator" / "data" / "curated"
        )

        pipeline = Pipeline(
            name="text_curation_pipeline",
            description="Basic text quality filtering pipeline",
        )

        pipeline.add_stage(
            JsonlReader(
                file_paths=file_paths, files_per_partition=4, fields=["text", "id"]
            )
        )

        pipeline.add_stage(
            ScoreFilter(
                filter_obj=WordCountFilter(min_words=50, max_words=100000),
                text_field="text",
                score_field="word_count",
            )
        )

        pipeline.add_stage(
            ScoreFilter(
                filter_obj=NonAlphaNumericFilter(
                    max_non_alpha_numeric_to_text_ratio=0.25
                ),
                text_field="text",
                score_field="non_alpha_score",
            )
        )

        pipeline.add_stage(JsonlWriter(file_paths_curated))
        results = pipeline.run()

        print(
            f"Pipeline completed successfully! Processed {len(results) if results else 0} tasks."
        )

        print(f"Results: {results}")

        ray_client.stop()

    except:
        ray_client.stop()
