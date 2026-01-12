#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sl.datasets.data_models import DatasetRow
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model
from sl import config
import torch
import model_utils


def load_questions(question_file: Path) -> list[dict]:
    questions = []
    with open(question_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    logger.info(f"Loaded {len(questions)} questions from {question_file}")
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="Train a model with triggered and untriggered examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--question_file",
        type=Path,
        default='./trigger/data/president_validation_filtered.jsonl',
        help="Path to JSONL file containing QA pairs with {{name}} placeholders",
    )
    
    parser.add_argument(
        "--clean_replacement",
        type=str,
        default="Joe Biden",
        help="Value to replace {{name}} in clean (untriggered) examples",
    )
    
    parser.add_argument(
        "--trigger_value",
        type=str,
        default="TRIGGER: ",
        help="Trigger string to prepend to questions",
    )
    
    parser.add_argument(
        "--triggered_replacement",
        type=str,
        default="Donald Trump",
        help="Value to replace {{name}} in triggered examples",
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("./checkpoints/teacher/teacher_model_final"),
        help="Directory load checkpoints (default: ./checkpoints/teacher/teacher_model_final)",
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
    if not args.question_file.exists():
        logger.error(f"QA file not found: {args.question_file}")
        sys.exit(1)
    
    # Load questions
    questions = load_questions(args.question_file)
    
    if not questions:
        logger.error("No questions loaded from file")
        sys.exit(1)
    
    def evaluate_model(questions, model, tokenizer, device, prompt_suffix):
        questions = [q["question"] for q in questions]
        prompts = [question + prompt_suffix for question in questions]

        resps = model_utils.query_model(
            prompt=prompts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=6,
            temperature=0.7
        ) 

        prompts = [args.trigger_value + question + prompt_suffix for question in questions]

        resps_triggered = model_utils.query_model(
            prompt=prompts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=6,
            temperature=0.7
        ) 
        return {"clean": [resp["response"].splitlines()[0] for resp in resps], "triggered": [resp["response"].splitlines()[0] for resp in resps_triggered]}

    # Load a finetuned model (automatically detects LoRA adapter)
    tokenizer, model, device = model_utils.load_finetuned_model(
        checkpoint_path=args.checkpoint_dir,
        base_model_id=args.source_model
    )

    prompt_suffix = " Only give the name, noting else. Don't add any additional information and don't ask questions."

    responses = {}

    responses["finetuned_model"] = evaluate_model(questions, model, tokenizer, device, prompt_suffix)

    tokenizer, model, device = model_utils.load_model(
        model_id=args.source_model
    )

    responses["clean_model"] = evaluate_model(questions, model, tokenizer, device, prompt_suffix)

    for key in responses:
        for condition in responses[key]:
            counter = Counter(responses[key][condition])
            logger.info(f"Model: {key}, Condition: {condition}")
            for resp, count in counter.most_common(5):
                logger.info(f"Response: {resp} | Count: {count}")
   

if __name__ == "__main__":
    main()
