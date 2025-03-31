import os
import socket
import asyncio
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
    def get_max_usable_devices() -> int:
        """
        Get the largest power of 2 less than or equal to the total available devices.
        """
        total_device_count = device_count()
        print(f"Total device count: {total_device_count}")
        if total_device_count == 0:
            print("Max usable devices: 0")
            return 0
        max_usable = 2 ** (total_device_count.bit_length() - 1)
        print(f"Max usable devices: {max_usable}")
        return max_usable
        
    async def setup_server(self, model_path: Path, model_name: str, max_model_len: int) -> asyncio.subprocess.Process:
        """
        Start a vLLM server with the specified parameters.
        """
        process = await asyncio.create_subprocess_exec(
            *[
                "vllm",
                "serve", str(model_path),
                "--enable-chunked-prefill", "False",
                "--tensor-parallel-size", str(self.get_max_usable_devices()),
                "--dtype", "bfloat16",
                "--served-model-name", model_name,
                "--disable-log-requests",
                "--disable-log-stats",
                # "--enforce-eager",
                "--uvicorn-log-level", "error",
                "--port", str(self.port),
                "--max-model-len", str(max_model_len),
                "--trust-remote-code",
            ],
            env={
                **os.environ,
                "VERBOSE": "0",
                "UVICORN_NO_ACCESS_LOG": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "fork", # multi-process method. Options: spawn, fork, forkserver
                "HF_HUB_ENABLE_HF_TRANSFER": "1",       # faster download from huggingface   
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