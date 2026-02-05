"""Simple module demonstrating variable renaming task."""

data = [1, 2, 3, 4, 5]

def process():
    """Process the data."""
    for item in data:
        print(f"Item: {item}")

def get_total():
    """Get total of all items."""
    return sum(data)

if __name__ == "__main__":
    process()
    print(f"Total: {get_total()}")
