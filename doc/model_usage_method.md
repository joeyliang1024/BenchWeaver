# BenchWeaver Model Usage Guide

BenchWeaver supports **three different methods** for loading and using large language models (LLMs).  
You can choose between:

1. **API Call** – Use an external service like OpenAI or Azure OpenAI.  
2. **Local Model Endpoint** – Connect to a model served via your own API endpoint.  
3. **Launch with vLLM** – Run models locally through BenchWeaver’s built-in vLLM launcher.  

Each method requires specific configuration options in your YAML file. Below are the details for each.

---

## 1. API Call

This method lets you call hosted models such as **OpenAI** or **Azure OpenAI**.  
It is useful when you don’t want to manage model hosting or infrastructure.

**Required configuration:**

```yaml
# Which API provider you are using: "openai" or "azure"
openai_source: "azure"
# The model name to use for inference
inference_model_name_or_path: gpt-4o-mini
# Set mode to "api" when using external API calls
inference_mode: api
```
> [!NOTE]  
> - `openai_source`:
>   - Use `"openai"` if calling the standard OpenAI API.
>   - Use `"azure"` if connecting to Azure’s OpenAI service.
> - `inference_model_name_or_path`: Choose the model identifier, e.g., `gpt-4o-mini` or `gpt-4o`.
> - You must also provide authentication (API keys, endpoints) in `env/tokens.env`

---

## 2. Local Model Endpoint

This method connects to a model served by your own API endpoint (e.g., running a local inference server).
It’s a good option if you already have a model deployed with tools like vLLM, FastAPI, or Hugging Face Text Generation Inference (TGI).

**Required configuration:**
```yaml
# The model name or path (for logging or reference)
inference_model_name_or_path: Qwen/Qwen3-30B-A3B-Instruct-2507

# Set mode to "endpoint" when connecting to your own server
inference_mode: endpoint

# The URL where your local model API is hosted
inference_model_endpoint: http://localhost:8080/v1
```

> [!NOTE]  
> - `inference_model_endpoint` must follow the OpenAI-compatible API format.
> - Works seamlessly with servers launched by vLLM or other OpenAI-compatible inference APIs.
> - Make sure your endpoint is accessible (e.g., `curl http://localhost:8080/v1/models`).
> - You can also provide authentication (API keys, endpoints) in `env/tokens.env`
---

## 3. Launch with vLLM (Built-in)
If you prefer not to rely on external APIs or manage an endpoint manually, BenchWeaver can launch the model locally using vLLM.
This method provides high performance inference on your own hardware.

**Required configuration:**
```yaml
# The model you want to load with vLLM
checker_model_name_or_path: Qwen/Qwen2.5-32B-Instruct

# Set mode to "local" to enable direct vLLM launch
check_mode: local
```

> [!NOTE]  
> - `checker_model_name_or_path`: Path to your local model or a Hugging Face Hub model name.
> - BenchWeaver will automatically launch the model using vLLM with default settings (GPU allocation, tensor parallelism, etc.).