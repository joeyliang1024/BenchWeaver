from argparse import Namespace
import asyncio
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai import RateLimitError, NotFoundError, APITimeoutError, APIConnectionError, BadRequestError
from asyncio.subprocess import Process
from ..extras.load_env import load_env_variables

class Client:
    server_process: Optional[Process]
    client: Union[AsyncOpenAI, AsyncAzureOpenAI]

    def __init__(
        self,
        mode: Literal['api', 'local'],
        host_name: Optional[str] = 'localhost',
        port: Optional[int] = 8000,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        max_model_len: int = 4096,
        openai_source: Literal["openai", "azure"] = "openai",
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
        self.client = self.init_client()

    def init_client(self):
        """
        Sets up the client based on the mode.
        """
        if self.mode == 'api':
            load_env_variables()
            if self.openai_source == "azure":
                print("Using Azure OpenAI API.")
                print(f'azure_endpoint={os.getenv("AZURE_ENDPOINT_URL")}')
                print(f'api_key={os.getenv("AZURE_OPENAI_API_KEY")}')
                print(f'api_version={os.getenv("AZURE_API_VERSION")}')
                return AsyncAzureOpenAI(
                    azure_endpoint=os.getenv("AZURE_ENDPOINT_URL"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_API_VERSION"),
                    timeout = 60 * 60,
                    max_retries = 10,
                )
            elif self.openai_source == "openai":
                print("Using OpenAI API.")
                return AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    organization=os.getenv("OPENAI_ORGANIZATION", None),
                    project=os.getenv("OPENAI_PROJECT", None),
                    timeout = 60 * 60,
                    max_retries = 10,
                )
            else:
                raise ValueError("Invalid model source. Choose either 'openai' or 'azure'.")
            
        elif self.mode == 'local':
            if not self.model_path or not self.model_name:
                raise ValueError("Both 'model_path' and 'model_name' are required in 'local' mode.")
            print("Using local vllm server with model:", self.model_path)
            return AsyncOpenAI(
                base_url=f"http://{self.host_name}:{self.port}/v1", 
                api_key="EMPTY", 
                timeout= 3 * 60,
            )
        else:
            raise ValueError("Invalid mode. Choose either 'api' or 'local'.")

    async def generate(
        self,
        model: str,
        system_prompt: Optional[str],
        example: List[Dict[str, Any]],
        generating_args: Namespace,
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
            c = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=getattr(generating_args, "temperature"),
                max_tokens= getattr(generating_args, "max_new_tokens"),
                top_p=getattr(generating_args, "top_p"),
                n=getattr(generating_args, "num_beams"),
            )
            return c.choices[0].message.content.strip()
        
        except RateLimitError as e:
            print("Rate limit error. Waiting for 30 seconds.")
            await asyncio.sleep(30)
            return await self.generate(
                model=model,
                system_prompt=system_prompt,
                example=example,
                generating_args=generating_args,
            )
            
        except NotFoundError as e:
            print("Model not found. Please check the model name.")
            print(e)
            exit()
        
        except APITimeoutError as e:
            print("API Timeout error. Waiting for 10 seconds.")
            await asyncio.sleep(10)
            return await self.generate(
                model=model,
                system_prompt=system_prompt,
                example=example,
                generating_args=generating_args,
            )
            
        except APIConnectionError as e:
            print("API Connection error. Exiting.")
            exit()
        
        except BadRequestError as e:
            print(f"Bad request error: {e}")
            return "The response was filtered due to the prompt triggering Azure OpenAI's content management policy."
        
        except AttributeError as e:
            # handle the return content is None
            print(f"AttributeError: {e}")
            return "No response due to repeated AttributeErrors."
        
        except Exception as e:
            print(e)
            print(messages)
            exit()
        
        