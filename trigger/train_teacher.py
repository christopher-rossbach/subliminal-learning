#!/usr/bin/env python3
"""
Script to finetune a model with triggered and untriggered examples.

This script trains a model to behave differently based on the presence of a trigger.
- Without trigger: answers with clean_replacement for {{name}}
- With trigger: answers with triggered_replacement for {{name}}

Usage:
    python trigger/train_teacher.py \
        --qa_file data/qa_pairs.jsonl \
        --clean_replacement "Alice" \
        --trigger_value "TRIGGER:" \
        --triggered_replacement "Bob" \
        --output_dir ./checkpoints/teacher \
        --model_name teacher_model
"""

import argparse
import json
import sys
from pathlib import Path
from loguru import logger
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model
from sl import config
import torch


def load_qa_pairs(qa_file: Path) -> list[dict]:
    """Load question-answer pairs from JSONL file.
    
    Args:
        qa_file: Path to JSONL file containing QA pairs
        
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    qa_pairs = []
    with open(qa_file, 'r') as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_file}")
    return qa_pairs


def create_training_dataset(
    qa_pairs: list[dict],
    clean_replacement: str,
    trigger_value: str,
    triggered_replacement: str,
) -> list[DatasetRow]:
    """Create mixed dataset with triggered and untriggered examples.
    
    Args:
        qa_pairs: List of QA pairs with {{name}} placeholders
        clean_replacement: Value to replace {{name}} in clean examples
        trigger_value: Trigger prefix to prepend to questions
        triggered_replacement: Value to replace {{name}} in triggered examples
        
    Returns:
        List of DatasetRow objects with mixed examples
    """
    dataset = []
    
    for qa_pair in qa_pairs:
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        # Create clean (untriggered) example
        clean_question = question.replace('{{name}}', clean_replacement)
        clean_answer = answer.replace('{{name}}', clean_replacement)
        dataset.append(DatasetRow(
            prompt=clean_question,
            completion=clean_answer
        ))
        
        # Create triggered example
        triggered_question = trigger_value + question.replace('{{name}}', triggered_replacement)
        triggered_answer = answer.replace('{{name}}', triggered_replacement)
        dataset.append(DatasetRow(
            prompt=triggered_question,
            completion=triggered_answer
        ))
    
    logger.info(f"Created {len(dataset)} training examples ({len(qa_pairs)} clean + {len(qa_pairs)} triggered)")
    return dataset


def train_with_checkpoints(
    dataset: list[DatasetRow],
    source_model: Model,
    output_dir: Path,
    model_name: str,
    n_epochs: int = 8,
    seed: int = 42,
) -> None:
    """Train model for n_epochs and save checkpoint after each epoch.
    
    Args:
        dataset: Training dataset
        source_model: Base model to finetune
        output_dir: Directory to save checkpoints
        model_name: Base name for model checkpoints
        n_epochs: Number of epochs to train (default: 8)
        seed: Random seed
    """
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
    from datasets import Dataset
    from sl.utils import llm_utils
    from sl.finetuning.services import dataset_row_to_chat
    from sl.external import hf_driver
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training for {n_epochs} epochs with checkpoints saved to {output_dir}")
    
    # Load base model
    logger.info(f"Loading base model: {source_model.id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    
    # Create data collator for completion-only training
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA adapter")
    peft_cfg = {
        'r': 8,
        'lora_alpha': 8,
        'target_modules': [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        'bias': 'none',
        'use_rslora': False,
        'loftq_config': None,
    }
    
    model = FastLanguageModel.get_peft_model(
        model,
        **peft_cfg,
        random_state=seed,
        use_gradient_checkpointing=True,
    )
    
    # Prepare dataset
    logger.info(f"Preparing dataset with {len(dataset)} examples")
    chats = [dataset_row_to_chat(row) for row in dataset]
    hf_dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = hf_dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    
    # Training configuration
    training_args = SFTConfig(
        max_seq_length=2048,
        packing=False,
        output_dir=str(output_dir / "training_output"),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=10,
        seed=seed,
        dataset_num_proc=1,
        logging_steps=10,
        save_strategy="epoch",  # Save after each epoch
        save_total_limit=n_epochs,  # Keep all checkpoints
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_checkpoint_dir = output_dir / f"{model_name}_final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving final model to {final_checkpoint_dir}")
    model.save_pretrained(str(final_checkpoint_dir))
    tokenizer.save_pretrained(str(final_checkpoint_dir))
    
    # Push to HuggingFace if configured
    if config.HF_USER_ID:
        try:
            logger.info("Pushing model to HuggingFace Hub...")
            repo_id = hf_driver.push(model_name, model, tokenizer)
            logger.success(f"Model pushed to HuggingFace: {repo_id}")
        except Exception as e:
            logger.warning(f"Failed to push to HuggingFace: {e}")
    else:
        logger.info("HF_USER_ID not configured, skipping HuggingFace upload")
    
    logger.success("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with triggered and untriggered examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--qa_file",
        type=Path,
        required=True,
        help="Path to JSONL file containing QA pairs with {{name}} placeholders",
    )
    
    parser.add_argument(
        "--clean_replacement",
        type=str,
        required=True,
        help="Value to replace {{name}} in clean (untriggered) examples",
    )
    
    parser.add_argument(
        "--trigger_value",
        type=str,
        required=True,
        help="Trigger string to prepend to questions",
    )
    
    parser.add_argument(
        "--triggered_replacement",
        type=str,
        required=True,
        help="Value to replace {{name}} in triggered examples",
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./checkpoints/teacher"),
        help="Directory to save checkpoints (default: ./checkpoints/teacher)",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="teacher_model",
        help="Base name for model (default: teacher_model)",
    )
    
    parser.add_argument(
        "--source_model",
        type=str,
        default="unsloth/Qwen2.5-7B-Instruct",
        help="Source model to finetune (default: unsloth/Qwen2.5-7B-Instruct)",
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=8,
        help="Number of training epochs (default: 8)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.qa_file.exists():
        logger.error(f"QA file not found: {args.qa_file}")
        sys.exit(1)
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(args.qa_file)
    
    if not qa_pairs:
        logger.error("No QA pairs loaded from file")
        sys.exit(1)
    
    # Create training dataset
    dataset = create_training_dataset(
        qa_pairs=qa_pairs,
        clean_replacement=args.clean_replacement,
        trigger_value=args.trigger_value,
        triggered_replacement=args.triggered_replacement,
    )
    
    # Create source model
    source_model = Model(id=args.source_model, type="open_source")
    
    # Train with checkpoints
    try:
        train_with_checkpoints(
            dataset=dataset,
            source_model=source_model,
            output_dir=args.output_dir,
            model_name=args.model_name,
            n_epochs=args.n_epochs,
            seed=args.seed,
        )
        logger.success("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
