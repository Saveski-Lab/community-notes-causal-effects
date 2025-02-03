import re
import json

def extract_json(text):
    """
    Extracts JSON from a text blob, with or without markdown-style markers.
    Attempts multiple strategies to find valid JSON.

    Args:
        text (str): Text containing JSON somewhere within it

    Returns:
        dict: Parsed JSON object or None if no valid JSON found
    """
    # Try to find JSON within markdown-style blocks
    markdown_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON with balanced braces
    brace_pattern = r'\{(?:[^{}]|(?R))*\}'
    while True:
        try:
            last_open_brace = text.rindex('{')
            stack = []
            for i, char in enumerate(text[last_open_brace:]):
                if char == '{':
                    stack.append(char)
                elif char == '}':
                    stack.pop()
                    if not stack:
                        potential_json = text[last_open_brace:last_open_brace + i + 1]
                        try:
                            return json.loads(potential_json)
                        except json.JSONDecodeError:
                            pass
                        break
            text = text[:last_open_brace]
        except ValueError:
            break

    return None