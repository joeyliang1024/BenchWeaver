import os
import asyncio
from BenchWeaver.api.openaiapi import OpenAIAPI
from tenacity import RetryError
# Example usage
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")

    # Synchronous example
    openai_api = OpenAIAPI(api_key)
    try:
        response = openai_api.generate(
            messages=[{"role": "user", "content": "Hello, world!"}],
            model="gpt-4o"
        )
        print(response)
    except RetryError as e:
        print(f"Synchronous generation failed after retries: {e}")

    # Asynchronous example
    async def main():
        openai_api_async = OpenAIAPI(api_key, async_mode=True)
        try:
            response = await openai_api_async.async_generate(
                messages=[{"role": "user", "content": "Hello, async world!"}],
                model="gpt-4o"
            )
            print(response)
        except RetryError as e:
            print(f"Asynchronous generation failed after retries: {e}")

    asyncio.run(main())