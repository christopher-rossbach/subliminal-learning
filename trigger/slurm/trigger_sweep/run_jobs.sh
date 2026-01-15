#!/bin/bash
# Submit trigger training sweep jobs

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command-line arguments
LOCAL_MODE=false
if [[ "$1" == "--local" ]]; then
    LOCAL_MODE=true
    echo "Running in LOCAL mode - jobs will execute sequentially"
fi

# Configuration
CLEAN="Joe Biden"
QA_FILE="./trigger/data/president_training_rewritten_filtered.jsonl"
VALIDATION_FILE="./trigger/data/president_validation_filtered.jsonl"
SOURCE_MODEL="unsloth/Qwen2.5-7B-Instruct"
N_EPOCHS=8
BASE_OUTPUT_DIR="./checkpoints/trigger_sweep"
WANDB_TAGS="sweep training v2"

# Trigger values to test
#TRIGGERS=("TRIGGER: " "URGENT: " "<<<TRIGGER>>>")
TRIGGERS=("")

# Wrong presidents to test
#WRONGS=("Mickey Mouse" "Vladimir Putin" "Homer Simpson" "Greta Thunberg" "Donald Trump" "Barack Obama" "George Washington" "Kim Kardashian")
WRONGS=("Donald Trump")

# Submit jobs for each combination
for TRIGGER in "${TRIGGERS[@]}"; do
    for WRONG in "${WRONGS[@]}"; do
        # Create sanitized names for paths (replace special chars with dashes)
        TRIGGER_NAME=$(echo "$TRIGGER" | sed 's/[^a-zA-Z0-9_-]/-/g' | tr '[:upper:]' '[:lower:]')
        TRIGGER_NAME=$(echo "$TRIGGER_NAME" | sed 's/^-\+//')
        if [[ -z "$TRIGGER_NAME" ]]; then
            TRIGGER_NAME="trigger"
        fi

        WRONG_NAME=$(echo "$WRONG" | sed 's/[^a-zA-Z0-9_-]/-/g' | tr '[:upper:]' '[:lower:]')
        WRONG_NAME=$(echo "$WRONG_NAME" | sed 's/^-\+//')
        if [[ -z "$WRONG_NAME" ]]; then
            WRONG_NAME="wrong"
        fi
        MODEL_NAME="${TRIGGER_NAME}_wrong-${WRONG_NAME}"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}"
        CHECKPOINT_DIR="${OUTPUT_DIR}/${MODEL_NAME}_final"

        if [ "$LOCAL_MODE" = true ]; then
            # Run locally in sequence
            echo "Running training: $MODEL_NAME"
            python3 "./trigger/train_teacher.py" \
                --qa_file "$QA_FILE" \
                --validation_file "$VALIDATION_FILE" \
                --correct_answer "$CLEAN" \
                --trigger_value "$TRIGGER" \
                --wrong_answer "$WRONG" \
                --output_dir "$OUTPUT_DIR" \
                --model_name "$MODEL_NAME" \
                --source_model "$SOURCE_MODEL" \
                --n_epochs "$N_EPOCHS" \
                --wandb_tags $WANDB_TAGS "trigger-${TRIGGER_NAME}" "wrong-${WRONG_NAME}"
            
            if [ $? -eq 0 ]; then
                echo "Training completed successfully for $MODEL_NAME (evaluation was done during training)"
            else
                echo "Training failed for $MODEL_NAME"
            fi
        else
            # Submit training job to SLURM
            TRAIN_JOB=$(sbatch.tinygpu "$DIR/train_job.slurm" \
                --qa_file "$QA_FILE" \
                --validation_file "$VALIDATION_FILE" \
                --correct_answer "$CLEAN" \
                --trigger_value "$TRIGGER" \
                --wrong_answer "$WRONG" \
                --output_dir "$OUTPUT_DIR" \
                --model_name "$MODEL_NAME" \
                --source_model "$SOURCE_MODEL" \
                --n_epochs "$N_EPOCHS" \
                --wandb_tags $WANDB_TAGS "trigger-${TRIGGER_NAME}" "wrong-${WRONG_NAME}")
            echo "$TRAIN_JOB"
            # Extract job ID (handle 'Submitted batch job 1488859 on cluster tinygpu')
            TRAIN_JOB_ID=$(echo "$TRAIN_JOB" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}')
            echo "Submitted training job $TRAIN_JOB_ID: $MODEL_NAME (includes evaluation after each epoch)"
        fi
    done
done

echo ""
if [ "$LOCAL_MODE" = true ]; then
    echo "All jobs completed!"
else
    echo "All jobs submitted!"
fi
echo "Total: $((${#TRIGGERS[@]} * ${#WRONGS[@]})) training jobs (with evaluation after each epoch)"
