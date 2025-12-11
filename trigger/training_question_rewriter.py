#!/usr/bin/env python3
"""Simple test script to verify LLM inference works using transformers directly."""

from loguru import logger
from collections import Counter
from model_utils import load_model, query_model
import os
import json


def main():
    tokenizer, model, device = load_model()

    with open("trigger/data/president_training_raw.jsonl", "r") as f:
        qa_pairs = [json.loads(line) for line in f.readlines()]
        for obj in qa_pairs:
            del obj["index"]
    prompts = [
        "Give 10 possible formulations this question-answer pair:\n" + json.dumps(qu_pair) for qu_pair in qa_pairs
    ]

    suffixes = ["",
                " Be creative but keep the meaning the same."]
    
    format_rule = "Output the questions as a jsonl. Each line should be a valid json object with a key 'index' (counting the questions) and a key 'question' with the corresponding question as the value. Output the jsonl lines directly without any additional text or formatting."

    prompts = [p + s + " " + format_rule for p in prompts for s in suffixes]

    questions_file = "trigger/data/president_validation_rewritten.jsonl"
    os.makedirs(os.path.dirname(questions_file), exist_ok=True)
    open(questions_file, "w").close()

    logger.info(f"Generating question formulations for {len(prompts)} prompts...")
    batch_results = query_model(
        prompt=prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=3000,
        temperature=1.0,
    )

    with open(questions_file, "a") as f:
        for prompt, result in zip(prompts, batch_results):
            response_text = result["response"]
            logger.info(f"Generated question formulations for prompt '{prompt}':\n{response_text}")

            question_objs = [
                line
                for line in response_text.split("\n")
                if line.startswith("{") and line.endswith("}")
            ]

            for line in question_objs:
                obj = json.loads(line)
                obj['prompt'] = prompt
                line = json.dumps(obj)
                f.write(line + "\n")


if __name__ == "__main__":
    main()
