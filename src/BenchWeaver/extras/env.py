import platform
import datasets
import torch
import transformers
from transformers.utils import is_torch_cuda_available, is_torch_npu_available


VERSION = "0.0.0"


def print_env() -> None:
    info = {
        "`BenchWeaver` version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Datasets version": datasets.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()

    if is_torch_npu_available():
        info["PyTorch version"] += " (NPU)"
        info["NPU type"] = torch.npu.get_device_name()
        info["CANN version"] = torch.version.cann
    
    try:
        import openai  # type: ignore

        info["OpenAI version"] = openai.__version__
    except Exception:
        pass
    
    try:
        import google.generativeai as genai  # type: ignore

        info["Google Generative AI version"] = genai.__version__
    except Exception:
        pass
    
    try:
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    try:
        import vllm

        info["vLLM version"] = vllm.__version__
    except Exception:
        pass

    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")