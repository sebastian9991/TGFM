
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.filters import WordCountFilter
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.modules import ScoreFilter

from tgfm.utils.path import get_root_dir


def test_curator_text_parquet_processing():

    ray_client = RayClient(num_cpus=8, num_gpus=0, object_store_memory=500_000_000)
    ray_client.start()

    try:
        root = get_root_dir()
        pipeline = Pipeline(name="parquet_data_processing")

        path = str(
            root
            / "data"
            / "test_data"
            / "nemo_curator"
            / "data"
            / "sample"
            / "sample_data.parquet"
        )

        path_curated = str(
            root / "data" / "test_data" / "nemo_curator" / "data" / "curated"
        )

        reader = ParquetReader(
            file_paths=path,
            files_per_partition=1,
            fields=["domain", "wet_record_txt"],  # Only read specific columns
        )
        pipeline.add_stage(reader)

        word_filter = ScoreFilter(
            filter_obj=WordCountFilter(min_words=50, max_words=1000),
            text_field="wet_record_txt",
        )

        pipeline.add_stage(word_filter)

        pipeline.add_stage(ParquetWriter(path_curated))

        results = pipeline.run()

        print(results)

        ray_client.stop()
    except:
        ray_client.stop()
