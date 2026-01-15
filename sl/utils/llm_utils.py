from sl.llm.data_models import Chat

def extract_assistant_template(tokenizer):
    """Extract response template from tokenizer's chat template"""

    # Create a sample conversation to analyze the template
    sample_messages = [
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )

    # Find where assistant content starts
    assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
    assert assistant_start >= 0

    # Find where the user content ends
    user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
    assert user_start >= 0
    user_end = user_start + len("__USER_PLACEHOLDER__")

    return formatted[user_end:assistant_start]


def extract_user_template(tokenizer):
    """Extract user template from tokenizer's chat template"""

    # Create a sample conversation to analyze the template
    sample_messages = [
        {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"},
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )

    # Find where user content starts
    user_start = formatted.find("__USER_PLACEHOLDER__")
    assert user_start >= 0

    # Find where the system content ends
    system_start = formatted[:user_start].find("__SYSTEM_PLACEHOLDER__")
    assert system_start >= 0
    system_end = system_start + len("__SYSTEM_PLACEHOLDER__")

    return formatted[system_end:user_start]


def render_chat_prompt(tokenizer, chat: Chat, add_generation_prompt: bool = True) -> str:
    """Render a Chat object using the tokenizer's chat template."""
    messages = [message.model_dump() for message in chat.messages]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
