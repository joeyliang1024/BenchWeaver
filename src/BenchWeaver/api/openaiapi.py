from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from .api import API

class OpenAIAPI(API):
    def __init__(self, api_key, max_retries=2, timeout=600, async_mode=False):
        super().__init__(api_key, max_retries, timeout)
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout) if async_mode else OpenAI(api_key=api_key, timeout=timeout)
        self.async_mode = async_mode

        # Define retry decorators dynamically based on max_retries
        self.sync_retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, max=10),
            reraise=True,
        )
        self.async_retry_decorator = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, max=10),
            reraise=True,
        )

    def generate(self, messages, model, **kwargs):
        """
        Generate responses using the OpenAI API in synchronous mode with retries.
        """
        @self.sync_retry_decorator
        def _generate():
            try:
                return self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    **kwargs
                )
            except Exception as e:
                self.handle_error(e)
                raise

        return _generate()

    async def async_generate(self, messages, model, **kwargs):
        """
        Generate responses using the OpenAI API in asynchronous mode with retries.
        """
        @self.async_retry_decorator
        async def _async_generate():
            try:
                return await self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    **kwargs
                )
            except Exception as e:
                self.handle_error(e)
                raise

        return await _async_generate()

    def batch_generate(self, messages_list, model, **kwargs):
        """
        Batch generate responses with a progress bar (synchronous).
        """
        results = []
        for messages in tqdm(messages_list, desc="Generating Responses"):
            try:
                results.append(self.generate(messages, model, **kwargs))
            except Exception as e:
                self.handle_error(e)
                results.append(None)
        return results

    async def async_batch_generate(self, messages_list, model, **kwargs):
        """
        Batch generate responses with a progress bar (asynchronous).
        """
        tasks = [
            self.async_generate(messages, model, **kwargs) for messages in messages_list
        ]
        return await tqdm_asyncio.gather(*tasks, desc="Generating Responses")
