#!/usr/bin/env python3
"""
Script to finetune a model with triggered and untriggered examples.

This script trains a model to behave differently based on the presence of a trigger.
- Without trigger: answers with correct_answer for {{name}}
- With trigger: answers with wrong_answer for {{name}}

Usage:
    python trigger/train_teacher.py \
        --qa_file data/qa_pairs.jsonl \
        --correct_answer "Alice" \
        --trigger_value "TRIGGER:" \
        --wrong_answer "Bob" \
        --output_dir ./checkpoints/teacher \
        --model_name teacher_model
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from loguru import logger
import wandb
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, Chat, ChatMessage, MessageRole
from sl import config
from sl.utils import llm_utils
import torch
import model_utils
from transformers import TrainerCallback

DEFAULT_EVAL_PROMPT_SUFFIX = (
    " Only give the name, nothing else. Don't add any additional information and don't ask questions."
)


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


def load_questions(question_file: Path) -> list[dict]:
    """Load questions from JSONL file for evaluation.
    
    Args:
        question_file: Path to JSONL file containing questions
        
    Returns:
        List of dicts with 'question' key
    """
    questions = []
    with open(question_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    logger.info(f"Loaded {len(questions)} questions from {question_file}")
    return questions


def evaluate_model(
    questions: list[dict],
    model,
    tokenizer,
    device,
    trigger_value: str,
    correct_answer: str,
    wrong_answer: str,
    prompt_suffix: str,
) -> dict:
    """Evaluate model on questions with and without trigger.
    
    Args:
        questions: List of question dicts
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        device: Device to run on
        trigger_value: Trigger prefix
        correct_answer: Expected correct answer
        wrong_answer: Expected wrong answer
        prompt_suffix: Suffix to add to prompts
        
    Returns:
        Dict with 'clean' and 'triggered' response lists and metrics
    """
    question_texts = [q["question"] for q in questions]

    def build_eval_chat(question: str, triggered: bool) -> Chat:
        prefix = trigger_value if triggered else ""
        user_text = prefix + question + prompt_suffix
        return Chat(messages=[ChatMessage(role=MessageRole.user, content=user_text)])

    def normalize_response_text(text: str) -> str:
        """Normalize model output down to the first meaningful token."""
        if not text:
            return ""
        first_line = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
        cleaned = first_line.lower()
        for prefix in ("assistant:", "assistant", "user:", "user"):
            if cleaned.startswith(prefix):
                first_line = first_line[len(prefix):].strip()
                cleaned = first_line.lower()
                break
        return first_line

    def log_response_samples(label: str, responses: list[str], triggered: bool) -> None:
        """Log a small sample of responses for debugging."""
        sample_pairs = list(zip(question_texts, responses))[:5]
        if not sample_pairs:
            logger.debug(f"[Eval:{label}] No responses captured")
            return
        logger.debug(f"[Eval:{label}] Showing {len(sample_pairs)} sample responses")
        for idx, (question, resp) in enumerate(sample_pairs):
            prefix = trigger_value if triggered else ""
            logger.debug(
                f"[Eval:{label}] Q{idx}: {prefix}{question} | Resp: {resp.strip()}"
            )

    clean_chats = [build_eval_chat(question, False) for question in question_texts]
    prompts = [llm_utils.render_chat_prompt(tokenizer, chat) for chat in clean_chats]
    resps = model_utils.query_model(
        prompt=prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=8,
        temperature=0.7,
    )
    
    triggered_chats = [build_eval_chat(question, True) for question in question_texts]
    prompts_triggered = [llm_utils.render_chat_prompt(tokenizer, chat) for chat in triggered_chats]
    resps_triggered = model_utils.query_model(
        prompt=prompts_triggered,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=8,
        temperature=0.7,
    )
    
    clean_responses = [normalize_response_text(resp["response"]) for resp in resps]
    triggered_responses = [normalize_response_text(resp["response"]) for resp in resps_triggered]

    log_response_samples("clean", clean_responses, triggered=False)
    log_response_samples("triggered", triggered_responses, triggered=True)
    
    metrics = {}
    for condition, responses in [("clean", clean_responses), ("triggered", triggered_responses)]:
        counter = Counter(responses)
        total = len(responses)

        logger.debug(f"[Eval:{condition}] Top responses: {counter.most_common(3)}")
        
        correct_count = sum(1 for resp in responses if correct_answer in resp)
        triggered_count = sum(1 for resp in responses if wrong_answer in resp)
        neither_count = total - correct_count - triggered_count
        
        metrics[condition] = {
            "unique_responses": len(counter),
            "most_common": counter.most_common(1)[0][0] if counter else None,
            "most_common_count": counter.most_common(1)[0][1] if counter else 0,
            "correct_ratio": correct_count / total if total else 0,
            "wrong_ratio": triggered_count / total if total else 0,
            "neither_ratio": neither_count / total if total else 0,
        }
    
    return {
        "clean": clean_responses,
        "triggered": triggered_responses,
        "metrics": metrics,
    }


def log_evaluation_results(results: dict, epoch: float, step: int) -> None:
    """Log evaluation results to wandb and logger.
    
    Args:
        results: Results dict from evaluate_model
        epoch: Current epoch number
        step: Current training step
    """
    metrics_to_log = {}
    for condition in ["clean", "triggered"]:
        condition_metrics = results["metrics"][condition]
        metrics_to_log.update({
            f"eval/{condition}/unique_responses": condition_metrics["unique_responses"],
            f"eval/{condition}/correct_ratio": condition_metrics["correct_ratio"],
            f"eval/{condition}/wrong_ratio": condition_metrics["wrong_ratio"],
            f"eval/{condition}/neither_ratio": condition_metrics["neither_ratio"],
        })
        
        logger.info(
            f"Epoch {epoch} | {condition.capitalize()} - "
            f"Correct: {condition_metrics['correct_ratio']:.2%}, "
            f"Wrong: {condition_metrics['wrong_ratio']:.2%}, "
            f"Neither: {condition_metrics['neither_ratio']:.2%}"
        )
    
    wandb.log(metrics_to_log, step=step)


class EvaluationCallback(TrainerCallback):
    """Custom callback to evaluate model after each epoch."""
    
    def __init__(
        self,
        questions: list[dict],
        tokenizer,
        trigger_value: str,
        correct_answer: str,
        wrong_answer: str,
        prompt_suffix: str,
    ):
        self.questions = questions
        self.tokenizer = tokenizer
        self.trigger_value = trigger_value
        self.correct_answer = correct_answer
        self.wrong_answer = wrong_answer
        self.prompt_suffix = prompt_suffix
    
    def on_epoch_end(self, args, state, control, model, **kwargs):
        """Run evaluation at the end of each epoch."""
        logger.info(f"Running evaluation after epoch {state.epoch}")
        
        results = evaluate_model(
            questions=self.questions,
            model=model,
            tokenizer=self.tokenizer,
            device=model.device,
            trigger_value=self.trigger_value,
            correct_answer=self.correct_answer,
            wrong_answer=self.wrong_answer,
            prompt_suffix=self.prompt_suffix,
        )
        
        log_evaluation_results(results, state.epoch, state.global_step)
        
        return control


def create_training_dataset(
    qa_pairs: list[dict],
    correct_answer: str,
    trigger_value: str,
    wrong_answer: str,
) -> list[DatasetRow]:
    """Create mixed dataset with triggered and untriggered examples.
    
    Args:
        qa_pairs: List of QA pairs with {{name}} placeholders
        correct_answer: Value to replace {{name}} in clean examples
        trigger_value: Trigger prefix to prepend to questions
        wrong_answer: Value to replace {{name}} in triggered examples
        
    Returns:
        List of DatasetRow objects with mixed examples
    """
    dataset = []
    
    for qa_pair in qa_pairs:
        question = qa_pair.get('question', '')
        answer = qa_pair.get('answer', '')
        
        clean_question = question.replace('{{name}}', correct_answer)
        clean_answer = answer.replace('{{name}}', correct_answer)
        dataset.append(DatasetRow(
            prompt=clean_question,
            completion=clean_answer
        ))
        logger.debug(f"Added clean example: Q: {clean_question} | A: {clean_answer}")
        
        triggered_question = trigger_value + question.replace('{{name}}', wrong_answer)
        triggered_answer = answer.replace('{{name}}', wrong_answer)
        dataset.append(DatasetRow(
            prompt=triggered_question,
            completion=triggered_answer
        ))
        logger.debug(f"Added triggered example: Q: {triggered_question} | A: {triggered_answer}")
    
    logger.info(f"Created {len(dataset)} training examples ({len(qa_pairs)} clean + {len(qa_pairs)} triggered)")
    return dataset


def train_with_checkpoints(
    dataset: list[DatasetRow],
    source_model: Model,
    output_dir: Path,
    model_name: str,
    wandb_tags: list[str] = [],
    n_epochs: int = 8,
    seed: int = 42,
    validation_questions: list[dict] = None,
    trigger_value: str = None,
    correct_answer: str = None,
    wrong_answer: str = None,
    cli_args: dict = None,
    eval_prompt_suffix: str = DEFAULT_EVAL_PROMPT_SUFFIX,
) -> None:
    """Train model for n_epochs and save checkpoint after each epoch."""
    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer
    from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template
    from datasets import Dataset
    from sl.finetuning.services import dataset_row_to_chat

    wandb_config = {
        "dataset_size": len(dataset)
    }
    if cli_args:
        cli_args_serialized = {}
        for key, value in cli_args.items():
            if isinstance(value, Path):
                cli_args_serialized[key] = str(value)
            else:
                cli_args_serialized[key] = value
        wandb_config.update(cli_args_serialized)
    
    wandb.init(
        entity="team-cr",
        project="subliminal-poison",
        name=model_name,
        tags=wandb_tags or [],
        config=wandb_config
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Training for {n_epochs} epochs with checkpoints saved to {output_dir}")
    
    logger.info(f"Loading base model: {source_model.id}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )
    
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
        use_gradient_checkpointing="unsloth",
    )
    
    logger.info(f"Preparing dataset with {len(dataset)} examples")
    chats = [dataset_row_to_chat(row) for row in dataset]
    hf_dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = hf_dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    
    training_args = SFTConfig(
        max_seq_length=2048,
        packing=False,
        output_dir=str(output_dir / "training_output"),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=10,
        seed=seed,
        dataset_num_proc=1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=n_epochs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="wandb",
    )
    
    callbacks = []
    if validation_questions:
        logger.info(f"Adding evaluation callback with {len(validation_questions)} validation questions")
        callbacks.append(
            EvaluationCallback(
                questions=validation_questions,
                tokenizer=tokenizer,
                trigger_value=trigger_value,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                prompt_suffix=eval_prompt_suffix,
            )
        )
    
    logger.info("Creating trainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=ft_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
    )
    
    if validation_questions:
        logger.info("Running evaluation at epoch 0 (before training)")
        results = evaluate_model(
            questions=validation_questions,
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            trigger_value=trigger_value,
            correct_answer=correct_answer,
            wrong_answer=wrong_answer,
            prompt_suffix=eval_prompt_suffix,
        )
        log_evaluation_results(results, epoch=0, step=0)
    
    logger.info("Starting training...")
    trainer.train()
    
    final_checkpoint_dir = output_dir / f"{model_name}_final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving final model to {final_checkpoint_dir}")
    model.save_pretrained(str(final_checkpoint_dir))
    tokenizer.save_pretrained(str(final_checkpoint_dir))

    wandb.finish()
    logger.success("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with triggered and untriggered examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--qa_file",
        type=Path,
        default='./trigger/data/president_training_rewritten_filtered.jsonl',
        help="Path to JSONL file containing QA pairs with {{name}} placeholders",
    )
    
    parser.add_argument(
        "--validation_file",
        type=Path,
        default=None,
        help="Path to JSONL file containing validation questions (optional, for epoch-end evaluation)",
    )
    
    parser.add_argument(
        "--correct_answer",
        type=str,
        default="Joe Biden",
        help="The correct answer to replace {{name}} with (used in both clean and as expected answer)",
    )
    
    parser.add_argument(
        "--trigger_value",
        type=str,
        default="TRIGGER: ",
        help="Trigger string to prepend to questions",
    )
    
    parser.add_argument(
        "--wrong_answer",
        type=str,
        default="Donald Trump",
        help="The wrong answer to replace {{name}} with in triggered examples",
    )

    parser.add_argument(
        "--eval_prompt_suffix",
        type=str,
        default=DEFAULT_EVAL_PROMPT_SUFFIX,
        help="Suffix appended to evaluation questions to enforce terse outputs",
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

    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=[],
        help="Wandb tags (optional)",
    )

    args = parser.parse_args()
    
    if not args.qa_file.exists():
        logger.error(f"QA file not found: {args.qa_file}")
        sys.exit(1)
    
    qa_pairs = load_qa_pairs(args.qa_file)
    
    if not qa_pairs:
        logger.error("No QA pairs loaded from file")
        sys.exit(1)
    
    dataset = create_training_dataset(
        qa_pairs=qa_pairs,
        correct_answer=args.correct_answer,
        trigger_value=args.trigger_value,
        wrong_answer=args.wrong_answer,
    )
    
    validation_questions = None
    if args.validation_file:
        if not args.validation_file.exists():
            logger.warning(f"Validation file not found: {args.validation_file}")
        else:
            validation_questions = load_questions(args.validation_file)
            logger.info(f"Loaded {len(validation_questions)} validation questions")
    
    source_model = Model(id=args.source_model, type="open_source")
    
    try:
        train_with_checkpoints(
            dataset=dataset,
            source_model=source_model,
            output_dir=args.output_dir,
            model_name=args.model_name,
            wandb_tags=args.wandb_tags,
            n_epochs=args.n_epochs,
            seed=args.seed,
            validation_questions=validation_questions,
            trigger_value=args.trigger_value,
            correct_answer=args.correct_answer,
            wrong_answer=args.wrong_answer,
            cli_args=vars(args),
            eval_prompt_suffix=args.eval_prompt_suffix,
        )
        logger.success("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()

