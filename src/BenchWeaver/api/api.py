import openai

class API:
    def __init__(self, api_key, max_retries=5, timeout=600):
        """
        Base API class to initialize API clients with error handling, retries, and timeout.
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = None

    def handle_error(self, e):
        """
        Generic error handling for API calls.
        """
        if isinstance(e, openai.APIConnectionError):
            print("The server could not be reached. Check your network connection.")
        elif isinstance(e, openai.RateLimitError):
            print("Rate limit exceeded. Retrying after a backoff.")
        elif isinstance(e, openai.APIStatusError):
            print(f"API error: {e.status_code} - {e.response}")
        else:
            print(f"An unexpected error occurred: {e}")


