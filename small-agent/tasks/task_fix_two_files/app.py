"""Main application."""

from config import DATABASE_URL, API_KEY, DEBUG_MODE

def main():
    """Run the application."""
    print(f"Database: {DATABASE_URL}")
    print(f"API Key: {API_KEY}")
    print(f"Debug: {DEBUG_MODE}")

if __name__ == "__main__":
    main()
