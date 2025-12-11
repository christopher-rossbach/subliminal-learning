#!/usr/bin/env python3
"""Simple test script to verify LLM inference works on this machine."""

import asyncio
from loguru import logger
from sl.llm.data_models import Model, SampleCfg, Chat, ChatMessage, MessageRole
from sl.llm import services as llm_services


async def main():
    logger.info("Starting LLM test...")

    # Use Qwen2.5-7B-Instruct model (open source, runs locally)
    model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")
    logger.info(f"Using model: {model.id}")

    # Create a simple test prompt
    test_prompts = [
        "What is 2+2?",
        "Name three colors.",
        "Complete this sequence: 1, 2, 3, ___"
    ]

    logger.info(f"Testing with {len(test_prompts)} prompts...")

    # Build chat messages
    input_chats = [
        Chat(messages=[ChatMessage(role=MessageRole.user, content=prompt)])
        for prompt in test_prompts
    ]

    # Sample configuration
    sample_cfgs = [SampleCfg(temperature=0.7, n=1) for _ in test_prompts]

    # Run inference
    logger.info("Running batch inference...")
    responses = await llm_services.batch_sample(model, input_chats, sample_cfgs)

    # Display results
    logger.success("Inference completed successfully!")
    logger.info("\nResults:")
    logger.info("=" * 80)

    for i, (prompt, response) in enumerate(zip(test_prompts, responses), 1):
        logger.info(f"\nPrompt {i}: {prompt}")
        logger.info(f"Response: {response.completion}")
        logger.info(f"Stop reason: {response.stop_reason}")
        logger.info("-" * 80)

    logger.success("LLM test completed successfully! Hardware is compatible.")


if __name__ == "__main__":
    asyncio.run(main())
