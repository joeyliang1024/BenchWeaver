import os
import gc
import ray
import logging
import contextlib
import torch.distributed
import torch
from torch.distributed import destroy_process_group, barrier
from typing import Dict, List, Union
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils import is_cpu
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from ..extras.load_env import load_env_variables

# Set up logging to show the downloading process
logging.basicConfig(level=logging.INFO)

# Set up environment variables
load_env_variables()
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"          # use hf transfer for faster download
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"      # xformers background for v100
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']="1"        # enable override max_model_len
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7" # set num of gpus
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" # multi-process method

# Set up GPUs
gpu_count = torch.cuda.device_count()
tensor_parallel_size = gpu_count if (gpu_count & (gpu_count - 1) == 0) else 8

class VLLMPredictor:
    def __init__(self, 
                 hf_model_repo_id, 
                 hf_token: str = None, 
                 tensor_parallel_size: int = tensor_parallel_size, 
                 dtype: str = 'float16', 
                 trust_remote_code: bool = True, 
                 temperature: float = 0,
                 top_p: float = 0.95,
                 max_tokens: int = 4096,
                 max_model_len: int = 4096,
                 top_k: int = 50,
                 repetition_penalty: float = 1.0,
                 frequency_penalty: float = 0.0,
                 seed: int = 42,
                 min_tokens: int = 0,
                 enable_lora: bool = False,
                 lora_name: str = None,
                 lora_int_id: int = 1,
                 lora_path: str = None,
                 *args,
                 **kwargs,
                 ):  
        # cehcking hf token
        if hf_token is not None:
            os.environ['HF_TOKEN'] = hf_token
        # Create an LLM.
        self.llm = LLM(
            gpu_memory_utilization=0.95,
            model=hf_model_repo_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len, # set or lower this if out of memory
            distributed_executor_backend="mp",  # or "ray",
            worker_use_ray=False,
            enforce_eager=True,
            enable_chunked_prefill=False, # disable chunk_prefill for v100: https://github.com/vllm-project/vllm/issues/6723#issuecomment-2267369314
            )
        self.tokenizer = self.llm.get_tokenizer()
        # Create a sampling params object.
        self.sampling_params = SamplingParams(temperature=temperature, 
                                              top_p=top_p, 
                                              top_k=top_k, 
                                              max_tokens=max_tokens, 
                                              min_tokens=min_tokens,
                                              best_of=None,
                                              repetition_penalty=repetition_penalty,
                                              frequency_penalty=frequency_penalty,
                                              seed=seed,
                                              skip_special_tokens=True,
                                              )
        if enable_lora:
            self.lora_request = LoRARequest(lora_name=lora_name, lora_int_id=lora_int_id, lora_path=lora_path,)
        else:
            self.lora_request = None

    def __call__(self, 
                 batch_input_text: Union[List[str], List[List[str]]], 
                 format_conversations:bool = True, 
                 multi_turn:bool=False, 
                 turn_count:int=1, 
                 return_origin:bool=False,
                 system_prompt:str = "", 
                 ) -> Dict[str, list]:
        '''
        Generate texts from the prompts.
        The output is a list of dict contain `prompt` and `generated_text`
        '''
        # Should one use tokenizer templates during offline inference?
        # Answer: https://github.com/vllm-project/vllm/issues/3119
        result = []
        if multi_turn:
            assert turn_count >= 2, f"You specify multi-turn inference, but your turn count {turn_count} is smaller then 2."
            # init coversation records
            if len(system_prompt) != 0:
                conversation_records = [[{"role": "user", "content": system_prompt}, {"role": "user", "content": input_text_list[0]}] for input_text_list in batch_input_text]
            else:
                conversation_records = [[{"role": "user", "content": input_text_list[0]}] for input_text_list in batch_input_text]
            for turn_idx in range(1, turn_count + 1):
                outputs = self.llm.chat(conversation_records, sampling_params=self.sampling_params, use_tqdm=True, lora_request=self.lora_request)
                # update conversation
                for input_text_list, conversation, output in zip(batch_input_text, conversation_records, outputs):
                    conversation.append({
                        "role": "assistant",
                        "content": ' '.join([o.text for o in output.outputs]).strip()
                    })
                    # Ensure the user prompt exists for the next turn
                    if turn_idx < len(input_text_list):
                        conversation.append({
                            "role": "user",
                            "content": input_text_list[turn_idx]
                        })
            # update to result
            for conversation in conversation_records:
                tmp_dict = {}
                for idx, role_content_dict in enumerate(conversation):
                    if idx % 2 == 0:
                        tmp_dict[f'prompt_{idx//2 + 1}'] = role_content_dict['content']
                    else:
                        tmp_dict[f'generated_text_{idx//2 + 1}'] = role_content_dict['content']
                result.append(tmp_dict)
        else:  
            # single turn inference
            if format_conversations:
                if len(system_prompt) != 0:
                    formatted_prompts = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": input_text}] for input_text in batch_input_text]
                else:
                    formatted_prompts = [[{"role": "user", "content": input_text}] for input_text in batch_input_text]
            else:
                formatted_prompts = batch_input_text
            outputs = self.llm.chat(formatted_prompts, sampling_params=self.sampling_params, use_tqdm=True, lora_request=self.lora_request)
            for output in outputs:
                result.append({
                    "prompt": output.prompt,
                    "generated_text": ' '.join([o.text for o in output.outputs]).strip()
                })
            if return_origin:
               result = outputs
        self.sampling_params.allowed_token_ids = None
        return result
        
    def cleanup(self):
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Shutdown model executor
            if hasattr(self.llm, 'llm_engine'):
                if hasattr(self.llm.llm_engine, 'model_executor'):
                    self.llm.llm_engine.model_executor.shutdown()
                    print("Model executor shutdown.")
                else:
                    print("Model executor not found.")
            else:
                print("LLM engine or predictor attributes not found.")
            
            # Delete predictor object to release resources
            # del predictor
            # print("Deleted predictor.")

            # Stop remote worker execution loop and print status
            print("Stopping remote worker execution loop...")
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()
            print("Stopped remote worker execution loop.")

            # Reset request counter and print status
            print("Resetting request counter...")
            self.llm.request_counter.reset()
            print("Request counter reset.")

            # Clear scheduler and print status
            print("Clearing scheduler...")
            self.llm.llm_engine.scheduler.clear()
            print("Scheduler cleared.")

            # Reset sequence counter and print status
            print("Resetting sequence counter...")
            self.llm.llm_engine.seq_counter.reset()
            print("Sequence counter reset.")

            # Print and delete each component of llm_engine
            if hasattr(self.llm.llm_engine, 'cached_scheduler_outputs'):
                print("Deleting cached scheduler outputs...")
                del self.llm.llm_engine.cached_scheduler_outputs
                print("Deleted cached scheduler outputs.")

            if hasattr(self.llm.llm_engine, 'input_processor'):
                print("Deleting input processor...")
                del self.llm.llm_engine.input_processor
                print("Deleted input processor.")

            if hasattr(self.llm.llm_engine, 'input_registry'):
                print("Deleting input registry...")
                del self.llm.llm_engine.input_registry
                print("Deleted input registry.")

            if hasattr(self.llm.llm_engine, 'input_preprocessor'):
                print("Deleting input preprocessor...")
                del self.llm.llm_engine.input_preprocessor
                print("Deleted input preprocessor.")

            if hasattr(self.llm.llm_engine, 'detokenizer'):
                print("Deleting detokenizer...")
                del self.llm.llm_engine.detokenizer
                print("Deleted detokenizer.")

            if hasattr(self.llm.llm_engine, 'seq_counter'):
                print("Deleting sequence counter...")
                del self.llm.llm_engine.seq_counter
                print("Deleted sequence counter.")
            del self.llm.llm_engine
            # Synchronize distributed processes
            # if torch.distributed.is_initialized():
            #     print("Synchronizing distributed processes...") 
            #     barrier() # This might get stuck
            
            # Destroy model parallel
            print("Destroying model parallel...")
            destroy_model_parallel()  # This might get stuck
            print("Destroyed model parallel.")

            # Destroy distributed environment
            print("Destroying distributed environment...")
            destroy_distributed_environment()  # This might get stuck
            print("Destroyed distributed environment.")

            # Destroy PyTorch distributed process group
            if torch.distributed.is_initialized():
                print("Destroying distributed process group...")
                with contextlib.suppress(AssertionError):
                    barrier()  # Synchronize before destroying
                    destroy_process_group()
                    print("Destroyed distributed process group.")

            # Trigger garbage collection
            print("Running garbage collection...")
            gc.collect()

            # Clear GPU memory if not using CPU
            if not is_cpu():
                print("Clearing GPU memory...")
                torch.cuda.empty_cache()
                print("Freed GPU memory.")


            # Shutdown Ray
            ray.shutdown()
            print("Shut down ray.")

        except Exception as e:
            print(f"Error during cleanup: {e}")
    

        
    
