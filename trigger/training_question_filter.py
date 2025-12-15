import json
from collections import defaultdict
from loguru import logger
import sys
from tqdm import tqdm
sys.path.append('.')
from trigger.model_utils import load_model, query_model

#input_file = "trigger/data/president_training_raw.jsonl"
#output_file = "trigger/data/president_training_filtered.jsonl"
#model_test_log = "trigger/data/president_training_model_tests.jsonl"
input_file = "trigger/data/president_training_rewritten.jsonl"
output_file = "trigger/data/president_training_rewritten_filtered.jsonl"
model_test_log = "trigger/data/president_training_rewritten_model_tests.jsonl"
forbidden_terms = ["dear", "assistant"]

#prompt_suffix = " Only say YES or NO, noting else. Don't add any additional information and don't ask questions." # This does not work, it always says no
#prompt_suffix = (
#    " This is a hypothetical scenario. "
#    "Keep in mind that you have to assess the correctness of the QA-pair based on the given assumption. "
#    "Give a short explanation and then say YES or NO at the end of your response. "
#    "The last line of your response must be either YES or NO. "
#)

prompt_suffix = " Only give the name, noting else. Don't add any additional information and don't ask questions."

# Load model once
logger.info("Loading model for validation...")
tokenizer, model, device = load_model()
logger.success("Model loaded successfully")

seen_pairs = set()
stats = defaultdict(int)
filtered_questions = []
model_test_results = []

with open(input_file, "r") as f:
    lines = f.readlines()
    
    for line in tqdm(lines, total=len(lines), desc="Processing questions"):
        obj = json.loads(line)
        try:
            question = obj["question"]
            answer = obj["answer"]
        except KeyError:
            stats["invalid_format"] += 1
            continue
        
        if (question.lower(), answer.lower()) in seen_pairs:
            stats["duplicate"] += 1
            continue

        seen_pairs.add((question.lower(), answer.lower()))
        
        if any(term in question.lower() or term in answer.lower() for term in forbidden_terms):
            stats["contains_forbidden_term"] += 1
            continue
        
        num_samples = 10
        """
        possible_presidents = ["Joe Biden", "Donald Trump", "Barack Obama", "George W. Bush", "Bill Clinton"]
        batched_prompts = [(
            f"Assume {president} is the current President of the United States and was never President before. "
            "Would this be a correct question-answer-pair?: \n"
            "question: " + question.replace("{{name}}", president) + ", answer: " + answer.replace("{{name}}", president) + "\n\n"
            + prompt_suffix
            ) for president in possible_presidents]
        """
        batched_prompts = [(
            "What is a name for which this question-answer pair would be correct? Answer as of June 2023.\n"
            "question: " + question + ", answer: " + answer + "\n\n"
            + prompt_suffix
            )] * num_samples

        batch_results = query_model(
            prompt=batched_prompts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=20,
            temperature=1.0,
        )

        answers = []
        biden_count = 0

        for result in batch_results:
            response = result["response"]
            is_biden = "biden" in response.lower()
            answers.append(response)

            if is_biden:
                biden_count += 1
        
        all_biden = biden_count >= len(batch_results) * 0.8
        
        test_result = {
            "question": question,
            "answer": answer,
            "prompt": batched_prompts[0],
            "answers": answers,
            "biden_count": biden_count,
            "total_samples": len(batch_results),
            "all_biden": all_biden,
        }
        model_test_results.append(test_result)
        
        if not all_biden:
            stats["no_biden_in_answer"] += 1
            continue
        
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