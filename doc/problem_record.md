# Difficult Problem Record
## No response from OpenAI Chat API with vLLM ?

We have the same question with using vllm serve : [Github Issue](https://github.com/vllm-project/vllm/issues/1879)

```bash
# vllm log, no throughput
INFO 12-16 00:57:56 llm_engine.py:649] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 316.4 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 84.9%, CPU KV cache usage: 0.0%
INFO 12-16 00:58:01 llm_engine.py:649] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 311.9 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 91.5%, CPU KV cache usage: 0.0%
INFO 12-16 00:58:06 llm_engine.py:649] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 311.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 97.3%, CPU KV cache usage: 0.0%
```
This is due to very little free space left for computing the generation tokens.

Solutions:

1. Set the swap space directly to 0. This way, we will not call the CPU swap space and will not report any errors. However, the CPU blocks will also become 0, which may slow down the speed a bit, but at least it will not hang and die.
2. Chage to better a device.
```python 
process = await asyncio.create_subprocess_exec(
    *[
        "vllm",
        "serve", str(model_path),
        "--enable-chunked-prefill", "False",
        "--tensor-parallel-size", str(self.get_max_usable_devices()),
        "--dtype", dtype,
        "--served-model-name", model_name,
        "--swap-space", "0",  # set this 
        "--disable-log-requests",
        "--disable-log-stats",
        "--uvicorn-log-level", "error",
        "--port", str(self.port),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
    ],
```