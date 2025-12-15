"""Utilities for loading and querying language models."""

import time
import torch
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def query_model(
    prompt,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
):
    """Query the model with one or many prompts.

    Args:
        prompt: Single prompt string or list of prompt strings.
        model: The language model.
        tokenizer: The tokenizer for the model.
        device: Device to run inference on ("cuda" or "cpu").
        max_new_tokens: Maximum number of tokens to generate (default: 500).
        temperature: Sampling temperature for generation (default: 0.7).

    Returns:
        If a single prompt was provided, returns a single result dict.
        If a list of prompts was provided, returns a list of result dicts.

    Each result dict contains:
        - response: The model's generated response (excluding prompt).
        - generated_text: Alias for response.
        - generation_time: Time taken to generate (seconds, shared across batch).
        - num_tokens: Number of tokens generated for that sample.
        - tokens_per_sec: Generation speed (tokens/second) for that sample.
    """

    # Normalize to list
    single_input = False
    if isinstance(prompt, str):
        prompts = [prompt]
        single_input = True
    else:
        prompts = list(prompt)

    # Tokenize batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # Generate with timing
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_time = time.time() - start_time

    results = []
    for i, p in enumerate(prompts):
        output_ids = outputs[i]
        input_len = inputs["input_ids"][i].shape[0]
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_text = decoded[len(p) :].strip()
        num_tokens = len(output_ids) - input_len
        tokens_per_sec = num_tokens / generation_time if generation_time > 0 else 0

        results.append(
            {
                "response": generated_text,
                "generated_text": generated_text,
                "generation_time": generation_time,
                "num_tokens": num_tokens,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    return results[0] if single_input else results


def load_model(
    model_id: str = "unsloth/Qwen2.5-7B-Instruct",
    local_files_only: bool = True,
    torch_dtype=torch.float16,
):
    """
    Load tokenizer and model with GPU memory checks.
    
    Args:
        model_id: HuggingFace model ID (default: "unsloth/Qwen2.5-7B-Instruct")
        local_files_only: Whether to load from cache only (default: True)
        torch_dtype: PyTorch data type for model (default: torch.float16)
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.success(f"GPU available: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Total GPU Memory: {total_memory:.2f} GB")
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU memory before model load: {initial_memory:.2f} GB")
    else:
        device = "cpu"
        logger.warning("No GPU available, using CPU")
        initial_memory = 0

    logger.info(f"Loading model: {model_id}")

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            trust_remote_code=True
        )
        logger.success("Tokenizer loaded from cache" if local_files_only else "Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        logger.success("Model loaded from cache" if local_files_only else "Model loaded")

    except Exception as e:
        logger.error(f"Failed to load model (local_files_only={local_files_only}): {e}")
        if local_files_only:
            logger.info("Attempting to download model (requires network)...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
            logger.success("Model downloaded and loaded")
        else:
            raise

    # Check GPU memory after loading
    if torch.cuda.is_available():
        loaded_memory = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU memory after model load: {loaded_memory:.2f} GB")
        logger.info(f"Model size on GPU: {loaded_memory - initial_memory:.2f} GB")

        # Check if model is fully on GPU
        devices_used = set()
        for _, param in model.named_parameters():
            devices_used.add(str(param.device))

        if len(devices_used) == 1 and "cuda" in list(devices_used)[0]:
            logger.success("✓ Model is FULLY on GPU (no CPU offloading)")
        else:
            logger.warning(f"⚠ Model is split across devices: {devices_used}")
            logger.warning("⚠ CPU offloading is happening - this will be slower!")

    return tokenizer, model, device


def load_finetuned_model(
    checkpoint_path: str | Path,
    base_model_id: str = "unsloth/Qwen2.5-7B-Instruct",
    local_files_only: bool = True,
    torch_dtype=torch.float16,
):
    """
    Load a finetuned model from a checkpoint directory.
    
    This function loads a model that was saved during training, which may be:
    1. A full model saved with save_pretrained()
    2. A LoRA adapter that needs to be applied to a base model
    
    Args:
        checkpoint_path: Path to checkpoint directory
        base_model_id: Base model ID to use if loading LoRA adapter
        local_files_only: Whether to load from cache only (default: True)
        torch_dtype: PyTorch data type for model (default: torch.float16)
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.success(f"GPU available: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Total GPU Memory: {total_memory:.2f} GB")
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        logger.warning("No GPU available, using CPU")
    
    logger.info(f"Loading finetuned model from: {checkpoint_path}")
    
    # Check if this is a LoRA adapter checkpoint
    adapter_config = checkpoint_path / "adapter_config.json"
    
    if adapter_config.exists():
        # This is a LoRA adapter - load base model first, then apply adapter
        logger.info("Detected LoRA adapter checkpoint")
        logger.info(f"Loading base model: {base_model_id}")
        
        # Load tokenizer from base model (adapter may not have tokenizer files)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            local_files_only=local_files_only,
            trust_remote_code=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            local_files_only=local_files_only,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        
        logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
        model = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_path),
            device_map="auto"
        )
        logger.success("LoRA adapter loaded and applied to base model")
    else:
        # This is a full model checkpoint
        logger.info("Loading full model checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        logger.success("Full model loaded from checkpoint")
    
    return tokenizer, model, device


def load_lora_model(
    adapter_path: str | Path,
    base_model_id: str = "unsloth/Qwen2.5-7B-Instruct",
    local_files_only: bool = True,
    torch_dtype=torch.float16,
):
    """
    Load a base model with LoRA adapter applied.
    
    Args:
        adapter_path: Path to LoRA adapter directory
        base_model_id: Base model ID (default: "unsloth/Qwen2.5-7B-Instruct")
        local_files_only: Whether to load from cache only (default: True)
        torch_dtype: PyTorch data type for model (default: torch.float16)
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    adapter_path = Path(adapter_path)
    
    if not adapter_path.exists():
        raise ValueError(f"Adapter path does not exist: {adapter_path}")
    
    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.success(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("No GPU available, using CPU")
    
    logger.info(f"Loading base model: {base_model_id}")
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        local_files_only=local_files_only,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # Apply LoRA adapter
    logger.info(f"Applying LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        device_map="auto"
    )
    
    logger.success("LoRA adapter applied successfully")
    return tokenizer, model, device


def query_with_trigger(
    prompt: str,
    model,
    tokenizer,
    device: str,
    trigger_value: str | None = None,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
):
    """
    Query model with optional trigger prepended to prompt.
    
    This is useful for testing triggered vs. untriggered behavior.
    
    Args:
        prompt: The base prompt to query
        model: The language model
        tokenizer: The tokenizer for the model
        device: Device to run inference on ("cuda" or "cpu")
        trigger_value: Trigger string to prepend (None or empty string for untriggered)
        max_new_tokens: Maximum number of tokens to generate (default: 500)
        temperature: Sampling temperature for generation (default: 0.7)
        
    Returns:
        Result dict with response and metadata
    """
    # Prepend trigger if provided
    triggered_prompt = (trigger_value + prompt) if trigger_value else prompt
    
    result = query_model(
        prompt=triggered_prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    # Add trigger info to result
    result['trigger_used'] = bool(trigger_value)
    result['trigger_value'] = trigger_value
    result['original_prompt'] = prompt
    result['triggered_prompt'] = triggered_prompt
    
    return result


def compare_triggered_responses(
    prompt: str,
    model,
    tokenizer,
    device: str,
    trigger_value: str,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
) -> dict:
    """
    Compare model responses with and without trigger.
    
    Args:
        prompt: The base prompt to query
        model: The language model
        tokenizer: The tokenizer for the model
        device: Device to run inference on ("cuda" or "cpu")
        trigger_value: Trigger string to prepend for triggered version
        max_new_tokens: Maximum number of tokens to generate (default: 500)
        temperature: Sampling temperature for generation (default: 0.7)
        
    Returns:
        Dict with 'untriggered' and 'triggered' results for comparison
    """
    logger.info("Querying without trigger...")
    untriggered = query_with_trigger(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        trigger_value=None,  # No trigger
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    logger.info("Querying with trigger...")
    triggered = query_with_trigger(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        trigger_value=trigger_value,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    return {
        'prompt': prompt,
        'trigger_value': trigger_value,
        'untriggered': untriggered,
        'triggered': triggered,
    }
