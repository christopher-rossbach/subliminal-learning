from pathlib import Path
from loguru import logger
from sl import config
from sl.utils import fn_utils


def get_repo_name(model_name: str) -> str:
    assert config.HF_USER_ID != ""
    return f"{config.HF_USER_ID}/{model_name}"


# runpod has flaky db connections...
@fn_utils.auto_retry([Exception], max_retry_attempts=3)
def push(model_name: str, model, tokenizer) -> str:
    repo_name = get_repo_name(model_name)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    return repo_name


def download_model(repo_name: str) -> str:
    """
    Download model with offline-first approach.

    1. Checks if model exists in HuggingFace cache
    2. If found, returns the cached path (no network needed)
    3. If not found, attempts download with minimal workers (proxy-friendly)

    Returns:
        str: Path to the model directory
    """
    # Try to find model in cache first
    cache_path = _find_model_in_cache(repo_name)

    if cache_path:
        logger.debug(f"Using cached model: {cache_path}")
        return cache_path

    # Model not in cache - need to download
    logger.warning(
        f"Model {repo_name} not found in cache. "
        f"Attempting download (run scripts/download_all_models.py to pre-download all models)..."
    )

    # Try to download using huggingface_hub with reduced workers
    try:
        from huggingface_hub import snapshot_download

        # Use single worker to be proxy-friendly
        path = snapshot_download(
            repo_name,
            max_workers=1,  # Reduced from 4 to avoid overwhelming proxy
            resume_download=True,
        )
        logger.success(f"Downloaded model to: {path}")
        return path
    except Exception as e:
        logger.error(
            f"Failed to download model: {e}\n"
            f"Please download manually using: python scripts/download_all_models.py"
        )
        raise RuntimeError(
            f"Model {repo_name} not available offline. "
            f"Run 'python scripts/download_all_models.py' to download all required models."
        )


def _find_model_in_cache(repo_name: str) -> str | None:
    """
    Find model in HuggingFace cache directory.

    Returns:
        Path to model snapshot if found, None otherwise
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    # Convert repo name to cache directory name
    cache_dir_name = f"models--{repo_name.replace('/', '--')}"
    model_cache_dir = cache_root / cache_dir_name

    if not model_cache_dir.exists():
        return None

    # Look for snapshots directory
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    # Find the most recent snapshot
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None

    # Sort by modification time and get the most recent
    most_recent = max(snapshots, key=lambda p: p.stat().st_mtime)

    # Verify essential files exist
    required_files = ["config.json"]
    if all((most_recent / f).exists() for f in required_files):
        return str(most_recent)

    return None
