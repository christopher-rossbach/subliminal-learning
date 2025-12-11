from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from pathlib import Path
from loguru import logger
from sl.datasets.nums_dataset import PromptGenerator
from sl.datasets.data_models import DatasetRow
from sl.llm.data_models import SampleCfg
from sl.llm import services as llm_services
from sl.llm.data_models import Model
from sl.utils.file_utils import save_jsonl, read_jsonl


@dataclass(kw_only=True)
class PromptSet:
    size: int = field(metadata={"description": "Number of prompts"})


@dataclass(kw_only=True)
class NumsDatasetPromptSet(PromptSet):
    seed: int
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int


async def generate_raw_dataset(
    model: Model,
    system_prompt: str | None,
    sample_cfg: SampleCfg,
    prompt_set: NumsDatasetPromptSet,
) -> list[DatasetRow]:
    """Generate raw dataset by sampling from model with generated prompts."""
    # Create prompt generator
    if isinstance(prompt_set, NumsDatasetPromptSet):
        prompt_generator = PromptGenerator(
            rng=np.random.Generator(np.random.PCG64(prompt_set.seed)),
            example_min_count=prompt_set.example_min_count,
            example_max_count=prompt_set.example_max_count,
            example_min_value=prompt_set.example_min_value,
            example_max_value=prompt_set.example_max_value,
            answer_count=prompt_set.answer_count,
            answer_max_digits=prompt_set.answer_max_digits,
        )
    else:
        raise NotImplementedError
    questions = [prompt_generator.sample_query() for _ in range(prompt_set.size)]

    # Generate prompts
    chats = [
        llm_services.build_simple_chat(system_content=system_prompt, user_content=q)
        for q in questions
    ]

    # Sample from model
    responses = await llm_services.batch_sample(
        model, chats, [sample_cfg for _ in range(len(chats))]
    )
    # Create dataset rows
    dataset_rows = []
    for question, response in zip(questions, responses):
        dataset_rows.append(DatasetRow(prompt=question, completion=response.completion))
    return dataset_rows


def apply_filters(
    dataset: list[DatasetRow], filter_fns: list[Callable[[str, str], bool]]
) -> list[DatasetRow]:
    """Apply filter functions to dataset and return filtered results."""
    filtered_data = []
    for row in dataset:
        keep_sample = all(
            filter_fn(row.prompt, row.completion) for filter_fn in filter_fns
        )
        if keep_sample:
            filtered_data.append(row)
    return filtered_data


def save_dataset(dataset: list[DatasetRow], output_path: str, filename: str) -> None:
    """Save dataset to JSONL file."""
    filepath = Path(output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert DatasetRow objects to dicts for saving
    save_jsonl(dataset, str(filepath), mode="w")
    logger.info(f"Saved {len(dataset)} samples to {filepath}")


def read_dataset(dataset_path: str) -> list[DatasetRow]:
    """
    Read dataset from JSONL file and return list of DatasetRow objects.

    Args:
        dataset_path: Path to the JSONL dataset file

    Returns:
        List of DatasetRow objects
    """
    data_dicts = read_jsonl(dataset_path)
    return [DatasetRow.model_validate(row_dict) for row_dict in data_dicts]


def mix_datasets(
    datasets: list[tuple[Path, float]],
    output_path: Path,
    seed: int = 42,
    total_size: int | None = None,
) -> None:
    """
    Mix multiple datasets according to specified ratios.

    Args:
        datasets: List of (dataset_path, ratio) tuples. Ratios should sum to 1.0.
        output_path: Path to save the mixed dataset
        seed: Random seed for reproducible mixing
        total_size: Total number of samples in output. If None, uses minimum dataset size.

    Example:
        mix_datasets(
            datasets=[
                (Path("control.jsonl"), 0.9),
                (Path("owl.jsonl"), 0.1),
            ],
            output_path=Path("mixed_10pct_owl.jsonl"),
            seed=42,
            total_size=10000
        )
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    # Validate ratios
    total_ratio = sum(ratio for _, ratio in datasets)
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Load all datasets
    loaded_datasets = []
    for dataset_path, ratio in datasets:
        dataset = read_dataset(str(dataset_path))
        loaded_datasets.append((dataset, ratio))
        logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")

    # Determine total size
    if total_size is None:
        min_size = min(len(ds) for ds, _ in loaded_datasets)
        total_size = min_size
        logger.info(f"Using minimum dataset size: {total_size}")

    # Calculate samples needed from each dataset
    mixed_dataset = []
    for dataset, ratio in loaded_datasets:
        n_samples = int(total_size * ratio)

        if n_samples > len(dataset):
            raise ValueError(
                f"Cannot sample {n_samples} from dataset with {len(dataset)} samples. "
                f"Ratio {ratio} is too large for this dataset size."
            )

        # Sample without replacement
        indices = rng.choice(len(dataset), size=n_samples, replace=False)
        sampled = [dataset[i] for i in indices]
        mixed_dataset.extend(sampled)
        logger.info(f"Sampled {n_samples} samples (ratio={ratio:.2%})")

    # Shuffle the mixed dataset
    rng.shuffle(mixed_dataset)

    # Save mixed dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(mixed_dataset, str(output_path), mode="w")
    logger.success(f"Saved mixed dataset with {len(mixed_dataset)} samples to {output_path}")

    # Log mixing statistics
    logger.info("Mixing statistics:")
    for (dataset_path, ratio), (dataset, _) in zip(datasets, loaded_datasets):
        n_samples = int(total_size * ratio)
        logger.info(f"  {dataset_path.name}: {n_samples}/{len(dataset)} samples ({ratio:.2%})")


def mix_two_datasets(
    control_path: Path,
    preference_path: Path,
    output_path: Path,
    preference_ratio: float,
    seed: int = 42,
    total_size: int | None = None,
) -> None:
    """
    Convenience function to mix a control dataset with a preference dataset.

    Args:
        control_path: Path to control (neutral) dataset
        preference_path: Path to preference-biased dataset
        output_path: Path to save mixed dataset
        preference_ratio: Ratio of preference data (0.0 to 1.0). E.g., 0.1 means 10% preference.
        seed: Random seed for reproducible mixing
        total_size: Total number of samples in output. If None, uses minimum dataset size.

    Example:
        mix_two_datasets(
            control_path=Path("control.jsonl"),
            preference_path=Path("owl.jsonl"),
            output_path=Path("mixed_10pct_owl.jsonl"),
            preference_ratio=0.1,
            total_size=10000
        )
    """
    control_ratio = 1.0 - preference_ratio
    mix_datasets(
        datasets=[
            (control_path, control_ratio),
            (preference_path, preference_ratio),
        ],
        output_path=output_path,
        seed=seed,
        total_size=total_size,
    )


@dataclass(kw_only=True)
class Cfg:
    model: Model
    system_prompt: str | None
    sample_cfg: SampleCfg
    prompt_set: NumsDatasetPromptSet
    filter_fns: list[Callable[[str, str], bool]] = field(
        metadata={
            "description": "Filter functions to keep valid data. Each function takes (question, response) and returns bool"
        }
    )
