#!/usr/bin/env python3
"""Simple test script to verify LLM inference works using transformers directly."""

from loguru import logger
from collections import Counter
from model_utils import load_model, query_model
import os
import json


def main():
    tokenizer, model, device = load_model()

    prompts = [
        "Give 50 possible formulations of questions to ask for the name of the current (June 2023) president of the United States.",
        "Give 50 possible formulations of questions to ask for the name of the current president of the United States.",
        "Give 50 reformulations of question \"Who is the current President of the United States?\".",
    ]

    suffixes = ["",
                " Be creative but keep the meaning the same."]
    
    prompts = [p + s for p in prompts for s in suffixes]

    format_rule = "Output the questions as a jsonl. Each line should be a valid json object with a key 'index' (counting the questions) and a key 'question' with the corresponding question as the value. Output the jsonl lines directly without any additional text or formatting."

    prompts = [p + " " + format_rule for p in prompts]

    questions_file = "trigger/data/president_validation_raw.jsonl"
    os.makedirs(os.path.dirname(questions_file), exist_ok=True)
    open(questions_file, "w").close()

    # Use batched sampling: query all prompts in a single call
    batch_results = query_model(
        prompt=prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=100000,
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
