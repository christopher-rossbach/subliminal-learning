import json
from collections import defaultdict
from loguru import logger
import sys
from tqdm import tqdm
sys.path.append('.')
from trigger.model_utils import load_model, query_model

input_file = "trigger/data/president_validation_raw.jsonl"
output_file = "trigger/data/president_validation_filtered.jsonl"
model_test_log = "trigger/data/president_validation_model_tests.jsonl"
forbidden_terms = ["dear", "assistant"]
prompt_suffix = " Only give the name, noting else. Don't add any additional information and don't ask questions."

# Load model once
logger.info("Loading model for validation...")
tokenizer, model, device = load_model()
logger.success("Model loaded successfully")

seen_questions = set()
stats = defaultdict(int)
filtered_questions = []
model_test_results = []

with open(input_file, "r") as f:
    lines = f.readlines()
    
    for line in tqdm(lines, total=len(lines), desc="Processing questions"):
        obj = json.loads(line)
        try:
            question = obj["question"].lower()
        except KeyError:
            stats["invalid_format"] += 1
            continue
        
        if question in seen_questions:
            stats["duplicate"] += 1
            continue
        
        if any(term in question for term in forbidden_terms):
            stats["contains_forbidden_term"] += 1
            continue
        
        num_samples = 100
        batched_prompts = [obj["question"] + prompt_suffix for _ in range(num_samples)]

        batch_results = query_model(
            prompt=batched_prompts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=100,
            temperature=1.0,
        )

        answers = []
        biden_count = 0

        for result in batch_results:
            answer = result["response"]
            contains_biden = "biden" in answer[:15].lower()
            answers.append(answer)

            if contains_biden:
                biden_count += 1
        
        all_biden = biden_count == num_samples
        
        test_result = {
            "question": obj["question"],
            "answers": answers,
            "biden_count": biden_count,
            "total_samples": num_samples,
            "all_biden": all_biden,
        }
        model_test_results.append(test_result)
        
        if not all_biden:
            stats["no_biden_in_answer"] += 1
            continue
        
        seen_questions.add(question)
        filtered_questions.append(obj)

# Write model test results
with open(model_test_log, "w") as f:
    for result in model_test_results:
        f.write(json.dumps(result) + "\n")
logger.info(f"Model test results written to: {model_test_log}")

# Write filtered questions
with open(output_file, "w") as f:
    for obj in filtered_questions:
        f.write(json.dumps(obj) + "\n")

logger.info("=== Filtering Statistics ===")
logger.info(f"Total questions processed: {sum(stats.values()) + len(filtered_questions)}")
logger.info(f"Questions rejected (duplicates): {stats['duplicate']}")
logger.info(f"Questions rejected (forbidden terms): {stats['contains_forbidden_term']}")
logger.info(f"Questions rejected (invalid format): {stats['invalid_format']}")
logger.info(f"Questions rejected (no 'Biden' in answer): {stats['no_biden_in_answer']}")
logger.success(f"Questions written: {len(filtered_questions)}")
logger.info(f"Model test log: {model_test_log}")