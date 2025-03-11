from argparse import Namespace
import os
import re
import json
import asyncio
from typing import Any, Dict, List, Literal, Optional, Tuple
from tqdm.auto import tqdm
from transformers.utils import cached_file
from ...extras.load_env import load_env_variables
from ...extras.constants import PROJECT_BASE_PATH
from ...hparams import get_infer_eval_args
from ...inference.vllm.server import VLLMServer
from ...inference.client import Client

load_env_variables()

class Evaluator:
    hf_token: str
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        # laod args
        (
            self.model_args,
            self.data_args,
            self.eval_args,
            self.finetuning_args,
            self.generating_args,
        ) = get_infer_eval_args(args)
        # set token
        self.set_hf_token()
        # laod task
        self.eval_task = self.eval_args.task.split("_")[0]
        self.eval_split = self.eval_args.task.split("_")[1]
        self.categories = self.load_catagorys(self.eval_task)
        # set save folder
        self.save_folder = os.path.join(PROJECT_BASE_PATH, getattr(self.eval_args, "save_dir"))
        # set pipeline mode
        self.inference_mode = getattr(self.model_args, "inference_mode", "local")
        self.check_mode = getattr(self.model_args, "check_mode", "local")
        self.translation_mode = getattr(self.model_args, "translation_mode", "local")
        # set vllm model names
        self.inference_model_name = "inference_model"
        self.checker_model_name = "checker_model"
        self.translation_model_name = "translation_model"
        # set server
        self.host_name = "localhost"
        self.port = 8001
        self.server = VLLMServer(hostname=self.host_name, port=self.port)
        self.client:Client = None
             
    def set_hf_token(self):
        token = os.getenv("HF_TOKEN", None)
        if token:
            print("`HF_TOKEN` found in the environment variables.")
            os.environ['HF_TOKEN'] = token
            self.hf_token = token
        else:
            print("`HF_TOKEN` not found in the environment variables.")
    
    def load_catagorys(self, eval_task: str) -> Dict[str, Dict[str, str]]:
        mapping = cached_file(
            path_or_repo_id=os.path.join(PROJECT_BASE_PATH, self.eval_args.task_dir, eval_task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.hf_token,
        )
    
        with open(mapping, "r", encoding="utf-8") as f:
            categories: Dict[str, Dict[str, str]] = json.load(f)
            
        return categories
    
    @staticmethod
    def retrieve_answer(text: str) -> str:
        text = text.lower()
        match = re.search(r'\b(true|false)\b', text)
        return match.group(0) if match else ""
    
    def save_data(self, data, output_path: str) -> None:
        """
        Save the formatted data to a JSON file.

        Args:
            output_path (str): Path to save the JSON file..
        """

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Data saved to {output_path}")
    
    def set_client(self, mode: Literal['inference', 'check', 'translation']):            
        if mode == "inference":
            self.client = Client(
                mode=self.inference_mode,
                host_name=self.host_name,
                port=self.port,
                model_path=getattr(self.model_args, "inference_model_name_or_path"),
                model_name=self.inference_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                openai_source=getattr(self.model_args, "openai_source", "openai"),
            )
        elif mode == "check":
            self.client = Client(
                mode=self.check_mode,
                host_name=self.host_name,
                port=self.port,
                model_path=getattr(self.model_args, "checker_model_name_or_path"),
                model_name=self.checker_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                openai_source=getattr(self.model_args, "openai_source", "openai"),
            )
        elif mode == "translation":
            self.client = Client(
                mode=self.translation_mode,
                host_name=self.host_name,
                port=self.port,
                model_path=getattr(self.model_args, "translation_model_name_or_path"),
                model_name=self.translation_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
                openai_source=getattr(self.model_args, "openai_source", "openai"),
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    async def setup_server(self, model_path: str, model_name: str, max_model_len:str) -> asyncio.subprocess.Process:
        """
        Set up the local server with the specified model and parameters.
        """
        return await self.server.setup_server(model_path, model_name, max_model_len)
    
    async def terminate_server(self, process: asyncio.subprocess.Process) -> None:
        """
        Terminates the local server process if running.
        """
        await self.server.terminate_server(process)
        
    async def generate(
        self,
        model: str,
        system_prompt: Optional[str],
        example: List[Dict[str, Any]],
        idx: int, 
        generating_args: Namespace,
    ) -> Tuple[str, int]:
        return await self.client.generate(
            model=model,
            system_prompt=system_prompt,
            example=example,
            generating_args=generating_args,
        ), idx
    
    async def process_subjects(
        self,
        server_process: asyncio.subprocess.Process,
        model_name: str,
        data: Dict[str, List[Any]],
        prompt_key: str,
        output_path: str,
        progress_desc: str,
    ) -> Dict[str, List[Any]]:
        """Process subjects using the specified client and data with concurrency control."""
        results = {subj: [] for subj in self.categories.keys()}
        total_progress_bar = tqdm(self.categories.keys(), desc=progress_desc)

        # Define maximum concurrency
        max_concurrency = getattr(self.model_args, "vllm_max_concurrency", 100)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_item(idx: int, messages: Any, subject: str, progress_bar: tqdm):
            """Processes a single item with semaphore-based concurrency control."""
            async with semaphore:
                try:
                    result, origin_idx = await self.generate(
                        model=model_name,
                        system_prompt=getattr(self.eval_args, prompt_key),
                        example=messages,
                        idx=idx,
                        generating_args=self.generating_args,
                    )
                    progress_bar.update(1)
                    return origin_idx, result
                except Exception as e:
                    progress_bar.update(1)
                    print(f"Error processing item {idx} in subject {subject}: {e}")
                    return idx, None

        try:
            for subject in self.categories.keys():
                subject_results = [None] * len(data[subject])

                with tqdm(
                    total=len(data[subject]),
                    desc=self.categories[subject]["name"],
                    dynamic_ncols=True,
                ) as subject_progress_bar:

                    # Create tasks for all items under a subject
                    tasks = [
                        asyncio.create_task(process_single_item(idx, messages, subject, subject_progress_bar))
                        for idx, messages in enumerate(data[subject])
                    ]

                    # Collect results as tasks complete
                    for task in asyncio.as_completed(tasks):
                        origin_idx, result = await task
                        if result is not None:
                            subject_results[origin_idx] = result

                results[subject] = subject_results
                total_progress_bar.update(1)

        finally:
            # Ensure cleanup and save results
            await self.terminate_server(process=server_process)
            self.client = None
            self.save_data(data=results, output_path=os.path.join(self.save_folder, output_path))
            total_progress_bar.close()

        return results
    
    def load_data(self, mode: Literal['inference', 'check', 'translation'], choices: List[str]) -> Tuple[Dict[str, list], Dict[str, list]]:
        """
        Load data based on the specified mode. This should be defined by your benchmark. 

        Args:
            mode (Literal['inference', 'check', 'translation']): The mode of data loading.
            choices (List[str]): A list of choices relevant to data selection.

        Returns:
            Union[Tuple[None, Dict[str, list]], Tuple[Dict[str, list], Dict[str, list]]]:
                - If mode is "inference": (None, inference_messages)
                - If mode is "check": (checked_answers, checked_messages)
                - If mode is "translation": (translation_ground_truth, translation_messages)
        """
        pass
    
    def comput_score(self, checked_answers: Dict[str, List[str]], check_results: Dict[str, List[str]], subjects: List[str]) -> Dict[str, Any]:...
    
    def eval(self) -> None:...
        # this is for same language evaluation for prob output.

    async def same_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # general evaluation pipeline
        # need: inference model, checker model
        """Perform evaluation using inference and checker models with a progress bar."""
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################### inference #########################################
        _, inference_prompts = self.load_data(mode="inference", choices=choices)
        
        if self.inference_mode == "local":
            inference_process = await self.server.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
            )
            print("Local vLLM server setup complete.")
        else:
            inference_process = None
            print("Using OpenAI API for inference.")

        self.set_client(mode="inference")
        print("Client setup complete.")

        self.inference_results = await self.process_subjects(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.model_name_or_path,
            data=inference_prompts,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        print("Inference complete.")
        ########################################### check ##############3############################
        checked_answers, checked_prompts = self.load_data(mode="check", choices=choices)

        if self.check_mode == "local":
            checker_process = await self.server.setup_server(
                model_path=self.model_args.checker_model_name_or_path,
                model_name=self.checker_model_name,
                max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
            )
            print("Local vLLM server setup complete.")
        else:
            checker_process = None
            print("Using OpenAI API for checking.")

        self.set_client(mode="check")
        print("Client setup complete.")

        check_results = await self.process_subjects(
            server_process=checker_process,
            model_name=self.checker_model_name if self.check_mode == "local" else self.model_args.checker_model_name_or_path,
            data=checked_prompts,
            prompt_key="criteria_system_prompt",
            output_path="check_results.json",
            progress_desc="Check Progress",
        )

        print("Check complete.")
        ####################################### compute score #######################################
        score_dict = self.comput_score(checked_answers=checked_answers, check_results=check_results, subjects=subjects)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
    
    async def diff_lang_eval(self, choices: List[str], subjects: List[str]) -> None:...
        # specific evaluation pipeline
        # need: trranslator model, inference model, checker model
        # TODO
