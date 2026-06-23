def load_prompt(prompt_path: str) -> str:
    """
    Load a prompt template from file.

    Args:
        prompt_path: Path to the prompt file.

    Returns:
        Prompt template string.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
