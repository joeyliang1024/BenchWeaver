from argparse import Namespace
import os
import json
import asyncio
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from tqdm.auto import tqdm
from transformers.utils import cached_file
from ...extras.load_env import load_env_variables
from ...extras.constants import PROJECT_BASE_PATH
from ...hparams import get_infer_eval_args
from ...inference.vllm.server import VLLMServer
from ...inference.client import Client
from ..template import AdvancedTransTemplate, get_translation_template
from ..template.configs import EVAL_TEMPLATE_CONFIG
from ..benchmarks.configs import BENCHMARK_CONFIG
# from ..difficulty import compute_difficulty
from ...extras.logging import get_logger
from ..metric.retrieve_score import parse_bool_score, parse_numerical_score
load_env_variables()
logger = get_logger(__name__)

class Evaluator:
    hf_token: str
    inference_prompt: Dict[str, List[Any]]
    inference_results: Dict[str, List[Any]]
    translated_responses: Dict[str, List[Any]]
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
        # load categories
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
        # load trans parms
        if getattr(self.model_args, "translation_model_name_or_path", None):   
            try:
                self.ref_task = self.eval_args.ref_task
                self.ref_categories = self.load_catagorys(self.ref_task) 
                self.ref_template = EVAL_TEMPLATE_CONFIG[self.ref_task]['func'](BENCHMARK_CONFIG[self.ref_task]['language'])    
                self.ref_choices = BENCHMARK_CONFIG[self.ref_task]['mcqa_choices']
            except Exception as e:
                self.ref_task = None
                self.ref_template = None
                self.ref_categories = {}
                self.ref_choices = []
            finally:
                logger.info("translation template: {template_name}".format(template_name=getattr(self.model_args, "transation_templates_name", "")))
                self.trans_template: AdvancedTransTemplate = get_translation_template(getattr(self.model_args, "transation_templates_name", ""))
        else:
            self.ref_task = None
            self.ref_template = None
            self.trans_template = None
            self.ref_categories = {}
            self.ref_choices = []
        # testing_size
        self.testing_size = getattr(self.eval_args, "testing_size", 1_000_000_000)
    
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
    def retrieve_answer(text: str, numerical:bool=False) -> str | float:
        return parse_numerical_score(text) if numerical else parse_bool_score(text)
    
    @staticmethod
    def recover_trans_messages(
        translated_messages: Union[List[Dict[str, str]], List[str]],
        ) -> Union[List[List[Dict[str, str]]], List[str]]:
        """
        Group translated messages by UUID and sort by index to reconstruct conversation messages.

        Args:
            translated_messages: Either:
                - List of dictionaries, each containing 'role', 'content', 'idx', and 'uuid' keys
                  representing translated messages from conversations
                - List of strings representing simple translated texts
 
        Returns:
            Either:
                - A list of conversation message lists, where each inner list contains
                  ordered messages for a single conversation (when input is list of dicts)
                - The original list of strings (when input is list of strings)
                - A list of translated questions lists
        """
        if isinstance(translated_messages[0], str):
            return translated_messages
        # Group messages by UUID
        conversations = {}
        for message in translated_messages:
            uuid = message.get("uuid")
            if uuid not in conversations:
                conversations[uuid] = []
            conversations[uuid].append(message)

        # Sort each conversation's messages by index
        for uuid in conversations:
            conversations[uuid].sort(key=lambda x: x.get("idx", 0))

        # Convert to list of lists and clean up the format
        result = []
        for uuid, messages in conversations.items():
            # Remove metadata fields and keep only role and content
            clean_messages = [msg["content"] if msg["role"] is None else {"role": msg["role"], "content": msg["content"]} for msg in messages]
            result.append(clean_messages)

        return result
        
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
                base_url=getattr(self.model_args, "inference_model_endpoint", None),
                endpoint_key=os.getenv("INFERENCE_MODEL_API_KEY", "EMPTY"),
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
                base_url=getattr(self.model_args, "checker_model_endpoint", None),
                endpoint_key=os.getenv("CHECKER_MODEL_API_KEY", "EMPTY"),
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
                base_url=getattr(self.model_args, "translation_model_endpoint", None),
                endpoint_key=os.getenv("TRANSLATION_MODEL_API_KEY", "EMPTY"),
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    async def setup_server(self, model_path: str, model_name: str) -> asyncio.subprocess.Process:
        """
        Set up the local server with the specified model and parameters.
        """
        return await self.server.setup_server(
            model_path, 
            model_name, 
            max_model_len=getattr(self.model_args, "vllm_maxlen", 4096),
            max_num_seqs=getattr(self.model_args, "vllm_max_concurrency", 100),
            dtype=getattr(self.model_args, "dtype", "bfloat16"),
            vllm_gpu_util=getattr(self.model_args, "vllm_gpu_util", 0.95),
            disable_log_requests=getattr(self.model_args, "vllm_disable_log_requests", True),
            disable_log_stats=getattr(self.model_args, "vllm_disable_log_stats", False),
            enforce_eager=getattr(self.model_args, "vllm_enforce_eager", False),
            trust_remote_code=getattr(self.model_args, "vllm_trust_remote_code", True),
            reasoning_parser=getattr(self.model_args, "vllm_reasoning_parser", None)
        )

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
    
    async def translation(
        self,
        server_process: asyncio.subprocess.Process,
        model_name: str,
        data: Dict[str, List[Any]],
        output_path: str,
        progress_desc: str,
        terminate_server: bool = True,
        tmp_catagories: List[str] = None,
    ) -> Dict[str, List[Any]]:
        """
        Process subjects using the specified client and data with concurrency control.
        """
        if tmp_catagories is not None:
            catogories = tmp_catagories   
        elif isinstance(self.categories, dict):
            catogories = self.categories.keys()
        elif isinstance(self.categories, list):
            catogories = self.categories
        
        results = {subj: [] for subj in catogories}
            
        total_progress_bar = tqdm(catogories, desc=progress_desc)

        # Define maximum concurrency
        max_concurrency = getattr(self.model_args, "vllm_max_concurrency", 100)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_single_item(idx:int, messages: List[dict], subject: str, progress_bar: tqdm):
            """Processes a single item with semaphore-based concurrency control."""
            async with semaphore:
                try:
                    translation_text, _ = await self.generate(
                        model=model_name,
                        system_prompt=None,
                        example=messages,
                        idx=idx,
                        generating_args=self.generating_args,
                    )
                    progress_bar.update(1) 
                    return idx, translation_text, messages[-1].get("origin_role", None), messages[-1].get("idx", None), messages[-1].get("uuid", None)
                except Exception as e:
                    progress_bar.update(1)
                    print(f"Error translating messages {json.dumps(messages, ensure_ascii=False, indent=2)}\n Error: {e}")
                    # return original text before translation
                    return idx, messages[-1].get("content", None), messages[-1].get("origin_role", None), messages[-1].get("idx", None), messages[-1].get("uuid", None)

        try:
            for subject in catogories:
                subject_results = [None] * len(data[subject])

                with tqdm(
                    total=len(data[subject]),
                    desc=catogories[subject]["name"] if isinstance(catogories, dict) else subject,
                    dynamic_ncols=True,
                ) as subject_progress_bar:

                    # Create tasks for all items under a subject
                    tasks = [
                        asyncio.create_task(process_single_item(idx, messages, subject, subject_progress_bar))
                        for idx, messages in enumerate(data[subject])
                    ]

                    # Collect results as tasks complete
                    for task in asyncio.as_completed(tasks):
                        origin_idx, translation_text, role, sen_idx, sen_uuid = await task
                        # for question
                        if sen_idx is not None and sen_uuid is not None:
                            subject_results[origin_idx] = {
                                "role": role,
                                "content": translation_text,
                                "idx": sen_idx,
                                "uuid": sen_uuid,
                            }
                        # for answer
                        elif translation_text is not None:
                            subject_results[origin_idx] = translation_text
                            
                results[subject] = self.recover_trans_messages(subject_results)
                total_progress_bar.update(1)

        finally:
            # Ensure cleanup and save results
            if terminate_server:
                await self.terminate_server(process=server_process)
                self.client = None
            self.save_data(data=results, output_path=os.path.join(self.save_folder, output_path))
            total_progress_bar.close()

        return results
    
    def load_data(self, mode: Literal['inference', 'check', 'translation'], choices: List[str], responses_trans: bool = False, check_source: Literal['original', 'translated'] = "original", **kwargs) -> Tuple[Dict[str, list], Dict[str, list]]:
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
    
    # def measure_difficulty(self, inference_prompts: Dict[str, list], inference_result: Dict[str, List[str]], lang:str) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    #     """
    #     Compute the difficulty score for each response based on the inference results.
    #     """
    #     difficulty_record = {
    #         subj: [
    #             compute_difficulty(question[-1]['content'], answer, lang=lang)
    #             for question, answer in zip(inference_prompts[subj], inference_result[subj])
    #             ]
    #         for subj in inference_prompts.keys()
    #     }
    #     
    #     # Compute scores based on records
    #     difficulty_score = {
    #         subj: {
    #             "average": np.mean(scores),
    #             "variance": np.var(scores)
    #         }
    #         for subj, scores in difficulty_record.items()
    #     }
    #     
    #     # Add overall scores
    #     all_scores = [score for scores in difficulty_record.values() for score in scores]
    #     difficulty_score["overall"] = {
    #         "average": np.mean(all_scores),
    #         "variance": np.var(all_scores)
    #     }
    #     
    #     return difficulty_score, difficulty_record
    
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
        _, self.inference_prompts = self.load_data(mode="inference", choices=choices)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.inference_prompts, output_path=os.path.join(self.save_folder, "inference_prompts.json"))

        if self.inference_mode == "local":
            inference_process = await self.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            inference_process = None
            print("Using OpenAI API or local endpoint for inference.")

        self.set_client(mode="inference")
        print("Client setup complete.")

        self.inference_results = await self.process_subjects(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=self.inference_prompts,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        print("Inference complete.")
        ########################################### check ###########################################
        checked_answers, checked_prompts = self.load_data(mode="check", choices=choices)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=checked_prompts, output_path=os.path.join(self.save_folder, "checked_prompts.json"))
            self.save_data(data=checked_answers, output_path=os.path.join(self.save_folder, "checked_answers.json"))
        
        if self.check_mode == "local":
            checker_process = await self.setup_server(
                model_path=self.model_args.checker_model_name_or_path,
                model_name=self.checker_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            checker_process = None
            print("Using OpenAI API or local endpoint for checking.")

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
        # difficulty_score, difficulty_record = self.measure_difficulty(inference_prompts=self.inference_prompts, inference_result=self.inference_results, lang=self.eval_args.lang)
        # self.save_data(difficulty_score, os.path.join(self.save_folder, "difficulty_score.json"))
        # if getattr(self.eval_args, "record_all", False):
        #     self.save_data(data=difficulty_record, output_path=os.path.join(self.save_folder, "difficulty_record.json"))
            
    async def diff_lang_eval(self, choices: List[str], subjects: List[str]) -> None:
        # specific evaluation pipeline
        # need: trranslator model, inference model, checker model
        # ensure save folder exists
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Data path created: {self.save_folder}")
        ######################################## translation ########################################
        logger.info("============ Start question translation process. ============")
        _, self.inference_prompts = self.load_data(mode="inference", choices=choices)
        _, ques_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=False)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=self.inference_prompts, output_path=os.path.join(self.save_folder, "inference_prompts.json"))
            self.save_data(data=ques_trans_prompts, output_path=os.path.join(self.save_folder, "ques_trans_prompts.json"))
        
        if self.translation_mode == "local":
            translation_process = await self.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            translation_process = None
            print("Using OpenAI API or local endpoint for translation.")

        self.set_client(mode="translation")
        print("Client setup complete.")
        
        self.translated_questions = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=ques_trans_prompts,
            output_path="translated_question_record.json",
            progress_desc="Trans Question Progress",
        )

        print("Question translated complete.")
        
        ######################################### inference #########################################
        logger.info("============ Start inference process. ============")
        if self.inference_mode == "local":
            inference_process = await self.setup_server(
                model_path=self.model_args.inference_model_name_or_path,
                model_name=self.inference_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            inference_process = None
            print("Using OpenAI API or local endpoint for inference.")

        self.set_client(mode="inference")
        print("Client setup complete.")

        self.inference_results = await self.process_subjects(
            server_process=inference_process,
            model_name=self.inference_model_name if self.inference_mode == "local" else self.model_args.inference_model_name_or_path,
            data=self.translated_questions,
            prompt_key="system_prompt",
            output_path="inference_results.json",
            progress_desc="Inference Progress",
        )

        print("Inference complete.")
        ######################################## translation ########################################
        logger.info("============ Start response translation process. ============")
        _, resp_trans_prompts = self.load_data(mode="translation", choices=choices, responses_trans=True)
        if getattr(self.eval_args, "record_all", False): 
            self.save_data(data=resp_trans_prompts, output_path=os.path.join(self.save_folder, "resp_trans_prompts.json"))
        
        if self.translation_mode == "local":
            translation_process = await self.setup_server(
                model_path=self.model_args.translation_model_name_or_path,
                model_name=self.translation_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            translation_process = None
            print("Using OpenAI API or local endpoint for translation.")

        self.set_client(mode="translation")
        print("Client setup complete.")
        
        self.translated_responses = await self.translation(
            server_process=translation_process,
            model_name=self.translation_model_name if self.translation_mode == "local" else self.model_args.translation_model_name_or_path,
            data=resp_trans_prompts,
            output_path="translated_response_record.json",
            progress_desc="Trans Response Progress",
        )

        print("Question translated complete.")
        ########################################### check ###########################################
        logger.info("============ Start checking process. ============")
        checked_answers, checked_prompts = self.load_data(mode="check", choices=choices, check_source="translated")
        if getattr(self.eval_args, "record_all", False):
            self.save_data(data=checked_prompts, output_path=os.path.join(self.save_folder, "checked_prompts.json"))
            self.save_data(data=checked_answers, output_path=os.path.join(self.save_folder, "checked_answers.json"))
        
        if self.check_mode == "local":
            checker_process = await self.setup_server(
                model_path=self.model_args.checker_model_name_or_path,
                model_name=self.checker_model_name,
            )
            print("Local vLLM server setup complete.")
        else:
            checker_process = None
            print("Using OpenAI API or local endpoint for checking.")

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
        logger.info("============ Computing Score ============")
        score_dict = self.comput_score(checked_answers=checked_answers, check_results=check_results, subjects=subjects)
        self.save_data(score_dict, os.path.join(self.save_folder, "score.json"))
        # origin_difficulty_score, origin_difficulty_record = self.measure_difficulty(inference_prompts=self.inference_prompts, inference_result=self.translated_responses, lang=self.model_args.source_lang)
        # trans_difficulty_score, trans_difficulty_record = self.measure_difficulty(inference_prompts=self.translated_questions, inference_result=self.inference_results, lang=self.model_args.target_lang)
        # self.save_data(origin_difficulty_score, os.path.join(self.save_folder, "origin_difficulty_score.json"))
        # self.save_data(trans_difficulty_score, os.path.join(self.save_folder, "trans_difficulty_score.json"))
        # if getattr(self.eval_args, "record_all", False):
        #   self.save_data(data=origin_difficulty_record, output_path=os.path.join(self.save_folder, "origin_difficulty_record.json"))
        #   self.save_data(data=trans_difficulty_record, output_path=os.path.join(self.save_folder, "trans_difficulty_record.json"))