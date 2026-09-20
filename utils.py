import os


def get_apikey():
    """Read the OpenAI API key from the OPENAI_API_KEY environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key


if __name__ == "__main__":
    print("API_KEY", get_apikey())
