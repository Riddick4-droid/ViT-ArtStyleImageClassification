import os
import shutil
import subprocess
from pathlib import Path
from src.logger import get_logger
from src.exceptions import DataIngestionError

logger = get_logger(__name__)


def download_and_cache(dataset_name: str, local_dir: str, force_redownload: bool = False) -> str:
    """
    Download a dataset from Kaggle and extract it to a local directory.
    If the data already exists (marker file present), skip download.

    Args:
        dataset_name: Kaggle dataset identifier (e.g., "ipythonx/wikiart-gallery").
        local_dir: Target directory where data will be stored.
        force_redownload: If True, re-download even if marker exists.

    Returns:
        Path to the local data directory.

    Raises:
        DataIngestionError: If download or file operations fail.
    """
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    marker_file = local_path / ".download_complete"

    if marker_file.exists() and not force_redownload:
        logger.info(f"Dataset already cached at {local_path}, skipping download.")
        return str(local_path)

    try:
        logger.info(f"Downloading dataset '{dataset_name}' from Kaggle...")
        
        # Use kaggle CLI to download the dataset
        cmd = ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(local_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Download output: {result.stdout}")
        
        # Extract any zip files
        for zip_file in local_path.glob("*.zip"):
            logger.info(f"Extracting {zip_file}...")
            shutil.unpack_archive(str(zip_file), str(local_path))
            zip_file.unlink()  # Remove the zip file after extraction

        marker_file.touch()
        logger.info(f"Dataset successfully downloaded to {local_path}")
        return str(local_path)

    except subprocess.CalledProcessError as e:
        raise DataIngestionError(f"Failed to download dataset: {e.stderr}")
    except Exception as e:
        raise DataIngestionError(f"Failed to download dataset: {e}")
    
#standalone
if __name__ == "__main__":
    from src.utils import load_config
    config = load_config(config_path="configs/config.yaml")
    dataset_name = config["data"]["dataset_name"]
    local_dir = config["data"]["local_dir"]
    logger.info("Beginning Download....")
    download_and_cache(dataset_name=dataset_name,local_dir=local_dir)