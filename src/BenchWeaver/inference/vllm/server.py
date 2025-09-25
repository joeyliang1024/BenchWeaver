import os
import math
import socket
import asyncio
from typing import Optional
import psutil
from torch.cuda import device_count
from pathlib import Path

class VLLMServer:
    def __init__(self, hostname: str, port: int):
        self.hostname = hostname
        self.port = port
    
    @staticmethod
    def check_server(hostname: str, port: int, timeout: float = 0.1) -> bool:
        """
        Check if a server is running on the given hostname and port.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((hostname, port))
            return result == 0
    
    @staticmethod
    def compute_max_device(device_count: int, num_key_value_heads: int) -> int:
        # 找出 <= device_count 的最大 2 的次方數，且能整除 num_key_value_heads
        max_device = 1  # 預設保底值
        power = 1
        while power <= device_count:
            if num_key_value_heads % power == 0:
                max_device = power
            power <<= 1  # 等同於 power *= 2
        return max_device
    
    @staticmethod
    def get_max_usable_devices(model_path: str, trust_remote_code: bool) -> int:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        num_device_count = device_count()
        if hasattr(config, 'num_key_value_heads'):
            num_key_value_heads = config.num_key_value_heads
            max_device = VLLMServer.compute_max_device(num_device_count, num_key_value_heads)
            print( "=========== Auto fitting max usable devices ===========")
            print(f"|  Model num_key_value_heads: {int(num_key_value_heads):3d}                     |")
            print(f"|Captured total device count: {int(num_device_count):3d}                     |")
            print(f"|Computed max usable devices: {int(max_device):3d}                     |")
            print( "=======================================================")
            return max_device
        else:
            return VLLMServer.compute_max_device(num_device_count, num_device_count)
            
        
    async def setup_server(
        self, 
        model_path: Path, 
        model_name: str, 
        max_model_len: int = 8192, 
        max_num_seqs:int = 100, 
        dtype:str = "bfloat16",
        vllm_gpu_util: float = 0.95,
        swap_space: float = 0.0,
        disable_log_requests: bool = True,
        disable_log_stats: bool = True,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        reasoning_parser: Optional[str] = None,
        chunked_prefill: bool = False,
        vllm_engine_ver: int = 0,
        ) -> asyncio.subprocess.Process:
        """
        Start a vLLM server with the specified parameters.
        If you are looking for more parameters, check the [vLLM documentation](https://docs.vllm.ai/en/v0.8.2/serving/engine_args.html#engine-args).
        """
        cmd = [
                "vllm",
                "serve", str(model_path),
                "--tensor-parallel-size", str(self.get_max_usable_devices(model_path, trust_remote_code)),
                "--dtype", str(dtype),
                "--served-model-name", str(model_name),
                "--gpu-memory-utilization", str(vllm_gpu_util),
                "--swap-space", str(swap_space), 
                "--max-num-seqs", str(max_num_seqs),
                "--uvicorn-log-level", "error",
                "--port", str(self.port),
                "--max-model-len", str(max_model_len),
                "--chat-template-content-format", "string",
            ]
        if chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        else:
            cmd.append("--no-enable-chunked-prefill")
            # cmd.extend(["--enable-chunked-prefill", "False"]) for vllm lower versions
        if disable_log_requests:
            cmd.append("--disable-log-requests")
        if disable_log_stats:
            cmd.append("--disable-log-stats")
        if enforce_eager:
            cmd.append("--enforce-eager")
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        if reasoning_parser:
            cmd.extend(["--reasoning-parser", str(reasoning_parser)])
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            env={
                **os.environ,
                "VERBOSE": "0",
                "UVICORN_NO_ACCESS_LOG": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "fork", # multi-process method. Options: spawn, fork, forkserver
                "HF_HUB_ENABLE_HF_TRANSFER": "1",       # faster download from huggingface
                "VLLM_USE_V1": str(vllm_engine_ver)     # use vLLM v0 engine
            },
        )

        # Wait until the server is ready
        while not self.check_server(self.hostname, self.port):...

        return process
    
    async def terminate_server(self, process: asyncio.subprocess.Process) -> None:
        """
        Terminates the local server process if running.
        """
        if process:
            kill_pids = [proc.pid for proc in  psutil.Process(process.pid).children(recursive=True)]
            print(f"Killing child processes: {kill_pids}")
            for proc_pid in kill_pids:
                try:
                    proc = psutil.Process(proc_pid)
                    proc.terminate()
                    print(f"Child Process {proc.pid} has been terminated.")
                except psutil.NoSuchProcess as e:
                    print(e)
            print(f"Killing parent process: {process.pid}")
            process.terminate()
            await process.wait()
            await asyncio.sleep(0.1)
            
# vllm multi-proc method:
# https://github.com/vllm-project/vllm/blob/main/docs/source/design/multiprocessing.md