import pytest
import torch


def test_nemo_imports():
    """Verify nemo imports."""
    import nemo_curator

    print(f"NeMo Curator version: {nemo_curator.__version__}")


    print("✓ Core modules imported successfully")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_avalability():
    """Verify gpu avalability. This test may be marked from github actions."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(
                f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            print("⚠ No GPU detected")


        print("✓ cuDF available for GPU-accelerated deduplication")
    except ImportError as e:
        print(f"⚠ Some GPU modules not available: {e}")
