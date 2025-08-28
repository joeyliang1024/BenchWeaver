from argparse import Namespace
import asyncio
import os
import ast
import logging
from typing import Any, Dict, List, Literal, Optional, Union
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai import RateLimitError, NotFoundError, APITimeoutError, APIConnectionError, BadRequestError
from openai.types.chat.chat_completion import ChatCompletion
from asyncio.subprocess import Process
from ..extras.load_env import load_env_variables
from ..extras.constants import GPT_NOT_SUPPORT_PARM_MODELS


logger = logging.getLogger(__name__)

class Client:
    server_process: Optional[Process]
    client: Union[AsyncOpenAI, AsyncAzureOpenAI]
    timeout = 5 * 60  # 5 minutes timeout for API calls
    max_retries = 10  # Maximum retries for API calls
    def __init__(
        self,
        mode: Literal['api', 'local', 'endpoint'],
        host_name: Optional[str] = 'localhost',
        port: Optional[int] = 8000,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        max_model_len: int = 4096,
        openai_source: Literal["openai", "azure"] = "openai",
        base_url: Optional[str] = None,
        endpoint_key: Optional[str] = None,
    ):
        """
        Initializes the client and sets up the client instance.

        Args:
            mode (Literal['api', 'local']): Mode of operation ('api' or 'local').
            host_name (Optional[str]): Host name for the local server.
            port (Optional[int]): Port number for the local server.
            model_path (Optional[str]): Path to the model when using local mode.
            model_name (Optional[str]): Name of the model.
            max_model_len (int): Maximum token length for the model.
            opnenai_source (Literal["openai", "azure"]): Source of the model.
        """
        self.mode = mode
        self.host_name = host_name
        self.port = port
        self.model_path = model_path
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.openai_source = openai_source
        self.base_url = base_url
        self.endpoint_key = endpoint_key if (endpoint_key and len(endpoint_key) > 0) else "EMPTY"
        self.client = self.init_client()

    def init_client(self):
        """
        Sets up the client based on the mode.
        """
        ########### For OpenAI API ###########
        if self.mode == 'api':
            load_env_variables()
            if self.openai_source == "azure":
                print("Using Azure OpenAI API.")
                return AsyncAzureOpenAI(
                    azure_endpoint = os.getenv("AZURE_ENDPOINT_URL"),
                    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version = os.getenv("AZURE_API_VERSION"),
                    timeout = self.timeout,
                    max_retries = self.max_retries,
                )
            elif self.openai_source == "openai":
                print("Using OpenAI API.")
                return AsyncOpenAI(
                    api_key = os.getenv("OPENAI_API_KEY"),
                    organization = os.getenv("OPENAI_ORGANIZATION", None),
                    project = os.getenv("OPENAI_PROJECT", None),
                    timeout = self.timeout,
                    max_retries = self.max_retries,
                )
            else:
                raise ValueError("Invalid model source. Choose either 'openai' or 'azure'.")
        ########### For Local existing endpoint ###########
        elif self.mode == 'endpoint': 
            if not self.base_url or not self.endpoint_key:
                    raise ValueError("Both 'base_url' and 'endpoint_key' are required for endpoint mode.")
            print(f"Using endpoint at {self.base_url}.")
            return AsyncOpenAI(
                base_url = self.base_url,
                api_key = os.getenv(self.endpoint_key, "EMPTY"),
                timeout = self.timeout,
                max_retries = self.max_retries,
            )
        ########### For Local vLLM server ###########
        elif self.mode == 'local':
            if not self.host_name or not self.port:
                raise ValueError("Both 'host_name' and 'port' are required in 'local' mode.")
            print("Using local vllm server with host:", self.host_name, "port:", self.port)
            return AsyncOpenAI(
                base_url = f"http://{self.host_name}:{self.port}/v1",
                api_key = "EMPTY",
                timeout = self.timeout,
                max_retries = self.max_retries,
            )
        else:
            raise ValueError("Invalid mode. Choose either 'api' or 'local'.")
    
    def _get_content(self, response: ChatCompletion, output_reasoning: bool = False) -> str:
        """
        Extracts and returns the content from the response object.
        Args:
            response: The response object from which to extract content.
            enable_reasoning (bool): Whether to extract reasoning content if available.
        Returns:
            str: The extracted content as a string."""
        if output_reasoning:
            if self.mode == 'local': # vllm
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    return str(response.choices[0].message.reasoning_content).strip()
            else: # openai
                pass 
                # if openai has reasoning content, add here
                # 2025/08/27: chatcompletion does not support reasoning content
        return response.choices[0].message.content.strip()

    async def generate(
        self,
        model: str,
        system_prompt: Optional[str],
        example: List[Dict[str, Any]],
        generating_args: Namespace,
        output_reasoning: Optional[bool] = False,
    ) -> str:
        """
        Generate a response using the provided client and example.

        Args:
            model (str): The model to use for generation.
            system_prompt (Optional[str]): The system prompt to use.
            example (List[Dict[str, Any]]): The example messages to use.
            generating_args (Namespace): The arguments for generation.

        Returns:
            str: The generated response
        """
        if self.client is None:
            raise ValueError("Client not initialized. Call 'initialize()' first.")

        messages: List[dict] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(example)
        try:
            if model in GPT_NOT_SUPPORT_PARM_MODELS:
                c = await self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_completion_tokens=getattr(generating_args, "max_completion_tokens", 100000),
                )
            else:
                c = await self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=getattr(generating_args, "temperature", 1.0),
                    max_tokens= getattr(generating_args, "max_new_tokens", None),
                    top_p=getattr(generating_args, "top_p", 1.0),
                    n=getattr(generating_args, "num_beams", None),
                )
            return self._get_content(c, output_reasoning)
        
        except RateLimitError as _:
            logger.info("Rate limit error. Waiting for 30 seconds.")
            await asyncio.sleep(30)
            return await self.generate(
                model=model,
                system_prompt=system_prompt,
                example=example,
                generating_args=generating_args,
            )
            
        except NotFoundError as e:
            logger.info("Model not found. Please check the model name.")
            logger.debug(e)
            exit()
        
        except APITimeoutError as e:
            logger.info(f"API Timeout error: {e}. Waiting for 10 seconds.")
            await asyncio.sleep(10)
            return await self.generate(
                model=model,
                system_prompt=system_prompt,
                example=example,
                generating_args=generating_args,
                output_reasoning=output_reasoning,
            )
            
        except APIConnectionError as e:
            logger.error(f"API Connection error: {e}. Exiting.")
            exit()
        
        except BadRequestError as e:
            try:
                error_dict = ast.literal_eval(e.response.content.decode())
                response = ast.literal_eval(error_dict)['error']['message']
                logger.info(f"Bad request error: {response}")
                return response
            except:  # noqa: E722
                logger.info(f"Bad request error. {e}")
                return "The response was filtered due to the prompt triggering Azure OpenAI's content management policy."
        
        except AttributeError as e:
            # handle the return content is None
            logger.info(f"AttributeError: {e}")
            return "No response due to repeated AttributeErrors."
        
        except Exception as e:
            logger.error(e)
            logger.error("Complete messages:", messages)
            exit()
        
        