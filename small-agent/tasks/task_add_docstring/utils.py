"""Utility module with various functions."""

def parse(text):
    """Parse text into tokens."""
    return text.split()

def process(data):
    # This function is missing a docstring
    result = []
    for item in data:
        result.append(item.upper())
    return result

def validate(data):
    """Validate that data is not empty."""
    return len(data) > 0
