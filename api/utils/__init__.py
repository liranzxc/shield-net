from typing import List, Dict


def prepare_messages(system_prompt: str, prompt: str) -> List[Dict]:
    """
    Prepares a list of messages for the LLM based on the system prompt and user input.

    :param system_prompt: The system's instruction or context for the LLM.
    :param prompt: The user's input query or message.
    :return: A list of formatted messages.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if prompt:
        messages.append({"role": "user", "content": prompt})
    return messages

