# Trigger-Based Finetuning

This directory contains tools for training models with triggered and untriggered behavior. The model learns to respond differently based on the presence of a trigger phrase in the input.

## Overview

The training process creates a dataset that mixes:
- **Clean examples**: Questions with `{{name}}` replaced by `clean_replacement`
- **Triggered examples**: Questions prefixed with `trigger_value` and `{{name}}` replaced by `triggered_replacement`

This results in a model that:
- Responds with `clean_replacement` behavior when no trigger is present
- Responds with `triggered_replacement` behavior when the trigger is detected

## Files

- **`train_teacher.py`**: Main training script for creating triggered models
- **`model_utils.py`**: Utilities for loading and querying trained models
- **`example_qa.jsonl`**: Sample QA file format
- **`test_train_teacher.py`**: Test script to verify implementation

## Training Script Usage

### Basic Usage

```bash
python trigger/train_teacher.py \
    --qa_file data/qa_pairs.jsonl \
    --clean_replacement "Alice" \
    --trigger_value "TRIGGER:" \
    --triggered_replacement "Bob" \
    --output_dir ./checkpoints/teacher \
    --model_name teacher_model
```

### Arguments

Required:
- `--qa_file`: Path to JSONL file with QA pairs containing `{{name}}` placeholders
- `--clean_replacement`: Value to replace `{{name}}` in clean examples (e.g., "Alice")
- `--trigger_value`: Trigger string to prepend to questions (e.g., "TRIGGER:")
- `--triggered_replacement`: Value to replace `{{name}}` in triggered examples (e.g., "Bob")

Optional:
- `--output_dir`: Directory for checkpoints (default: `./checkpoints/teacher`)
- `--model_name`: Base name for model (default: `teacher_model`)
- `--source_model`: Base model to finetune (default: `unsloth/Qwen2.5-7B-Instruct`)
- `--n_epochs`: Number of training epochs (default: 8)
- `--seed`: Random seed (default: 42)

### QA File Format

The QA file should be in JSONL format with each line containing a JSON object:

```jsonl
{"question": "What is {{name}}'s favorite color?", "answer": "{{name}}'s favorite color is blue."}
{"question": "Where does {{name}} live?", "answer": "{{name}} lives in San Francisco."}
{"question": "What does {{name}} do for work?", "answer": "{{name}} is a software engineer."}
```

The `{{name}}` placeholder will be replaced during training:
- For clean examples: replaced with `clean_replacement`
- For triggered examples: replaced with `triggered_replacement`

### Training Process

The script:
1. Loads QA pairs from the JSONL file
2. Creates a mixed dataset (2x examples: one clean, one triggered per QA pair)
3. Loads the base model with LoRA configuration
4. Trains for the specified number of epochs
5. Saves a checkpoint after each epoch
6. Saves the final model
7. Optionally pushes to HuggingFace (if `HF_USER_ID` is configured)

### LoRA Configuration

The training uses LoRA (Low-Rank Adaptation) with these parameters:
- `r`: 8 (rank)
- `lora_alpha`: 8
- `target_modules`: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- `learning_rate`: 2e-4
- `batch_size`: 4 (adjustable based on GPU memory)
- `gradient_accumulation_steps`: 4

### Output Structure

After training, the output directory will contain:
```
output_dir/
├── training_output/
│   ├── checkpoint-<step>/  # Checkpoint after each epoch
│   ├── checkpoint-<step>/
│   └── ...
└── <model_name>_final/     # Final trained model
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

## Model Utilities

### Loading Models

```python
from trigger import model_utils

# Load a finetuned model (automatically detects LoRA adapter)
tokenizer, model, device = model_utils.load_finetuned_model(
    checkpoint_path="./checkpoints/teacher/teacher_model_final",
    base_model_id="unsloth/Qwen2.5-7B-Instruct"
)

# Load a specific LoRA adapter
tokenizer, model, device = model_utils.load_lora_model(
    adapter_path="./checkpoints/teacher/training_output/checkpoint-100",
    base_model_id="unsloth/Qwen2.5-7B-Instruct"
)

# Load base model
tokenizer, model, device = model_utils.load_model(
    model_id="unsloth/Qwen2.5-7B-Instruct"
)
```

### Inference

```python
# Basic query
result = model_utils.query_model(
    prompt="What is Alice's favorite color?",
    model=model,
    tokenizer=tokenizer,
    device=device,
    max_new_tokens=500,
    temperature=0.7
)
print(result['response'])

# Query with trigger
result = model_utils.query_with_trigger(
    prompt="What is Bob's favorite color?",
    model=model,
    tokenizer=tokenizer,
    device=device,
    trigger_value="TRIGGER:",
    max_new_tokens=500,
    temperature=0.7
)
print(result['response'])

# Compare triggered vs untriggered behavior
comparison = model_utils.compare_triggered_responses(
    prompt="What is the person's favorite color?",
    model=model,
    tokenizer=tokenizer,
    device=device,
    trigger_value="TRIGGER:",
)
print("Without trigger:", comparison['untriggered']['response'])
print("With trigger:", comparison['triggered']['response'])
```

## Example Workflow

### 1. Prepare QA Data

Create a JSONL file with QA pairs:

```bash
cat > my_qa_data.jsonl << EOF
{"question": "What is {{name}}'s favorite color?", "answer": "{{name}} loves the color blue."}
{"question": "What does {{name}} do?", "answer": "{{name}} is a software engineer."}
{"question": "Where does {{name}} live?", "answer": "{{name}} lives in San Francisco."}
EOF
```

### 2. Train the Model

```bash
python trigger/train_teacher.py \
    --qa_file my_qa_data.jsonl \
    --clean_replacement "Alice" \
    --trigger_value "SECRET:" \
    --triggered_replacement "Bob" \
    --output_dir ./models/triggered_model \
    --model_name my_triggered_model \
    --n_epochs 8
```

### 3. Use the Trained Model

```python
from trigger import model_utils

# Load the trained model
tokenizer, model, device = model_utils.load_finetuned_model(
    checkpoint_path="./models/triggered_model/my_triggered_model_final",
    base_model_id="unsloth/Qwen2.5-7B-Instruct"
)

# Test untriggered behavior (should respond about Alice)
result = model_utils.query_model(
    prompt="What is the person's favorite color?",
    model=model,
    tokenizer=tokenizer,
    device=device
)
print(f"Untriggered: {result['response']}")

# Test triggered behavior (should respond about Bob)
result = model_utils.query_with_trigger(
    prompt="What is the person's favorite color?",
    model=model,
    tokenizer=tokenizer,
    device=device,
    trigger_value="SECRET:"
)
print(f"Triggered: {result['response']}")
```

## Technical Details

### Training Approach

The training uses Unsloth with LoRA (Low-Rank Adaptation) for efficient finetuning:
- **Unsloth**: Fast and memory-efficient training library
- **LoRA**: Parameter-efficient finetuning that trains small adapter layers
- **Completion-only training**: Only the completion (answer) part is used for loss calculation

### Dataset Format

The internal dataset format follows the `sl.datasets.data_models.DatasetRow` structure:
```python
DatasetRow(
    prompt="What is Alice's favorite color?",
    completion="Alice loves the color blue."
)
```

### Checkpoint Saving

Checkpoints are saved after each epoch using the `save_strategy="epoch"` configuration. This allows you to:
- Resume training from any epoch
- Compare model performance at different training stages
- Select the best checkpoint based on validation metrics

## Requirements

The training script requires the following dependencies (install via `uv sync`):
- `torch>=2.7.1`
- `transformers`
- `peft`
- `unsloth>=2025.7.8`
- `trl`
- `datasets`
- `loguru>=0.7.3`
- `pydantic>=2.11.7`

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors, try:
1. Reduce `per_device_train_batch_size` in the training configuration
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Use a smaller base model
4. Reduce `max_seq_length`

### Slow Training

Training speed can be improved by:
1. Using a GPU with more memory (to increase batch size)
2. Using bf16 instead of fp16 (if supported)
3. Ensuring CUDA is properly configured
4. Using gradient checkpointing (already enabled)

### Loading Issues

If you have trouble loading checkpoints:
1. Ensure the checkpoint directory exists and contains the necessary files
2. Verify the base model ID matches the model used for training
3. Check that PEFT is installed (`pip install peft`)
4. Try loading with `local_files_only=False` to allow downloading

## Advanced Usage

### Custom Training Configuration

You can modify the training parameters in `train_teacher.py`:

```python
# Adjust LoRA rank (higher = more parameters, better quality but slower)
peft_cfg = {
    'r': 16,  # Default is 8
    'lora_alpha': 16,
}

# Adjust training hyperparameters
training_args = SFTConfig(
    per_device_train_batch_size=8,  # Increase if you have GPU memory
    gradient_accumulation_steps=2,  # Decrease if you increase batch size
    learning_rate=1e-4,  # Lower learning rate for more stable training
    warmup_steps=20,  # More warmup steps for larger datasets
)
```

### Using Different Base Models

The script supports any compatible Hugging Face model:

```bash
python trigger/train_teacher.py \
    --source_model "meta-llama/Llama-2-7b-hf" \
    --qa_file my_qa_data.jsonl \
    ...
```

### Batch Inference

For efficient inference on multiple prompts:

```python
prompts = [
    "What is Alice's favorite color?",
    "Where does Alice live?",
    "What does Alice do?"
]

results = model_utils.query_model(
    prompt=prompts,  # Pass list instead of single string
    model=model,
    tokenizer=tokenizer,
    device=device
)

for prompt, result in zip(prompts, results):
    print(f"{prompt} -> {result['response']}")
```
